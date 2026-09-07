from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboDPAttentionPreparer
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.environ import envs
from sglang.srt.layers.cp.utils import get_cp_strategy
from sglang.srt.layers.dp_attention import world_dp_gather_enabled
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components.recv_skipper import (
    SchedulerRecvSkipper,
)
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.kv_cache_builder import uses_ssm_state
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
    cuda_graph_fully_disabled,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner import PrefillCudaGraphRunner
from sglang.srt.observability.metrics_collector import DPCooperationInfo
from sglang.srt.runtime_context import (
    get_exec,
    get_memory,
    get_parallel,
    get_schedule,
    get_spec,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import require_mlp_tp_gather

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.model_runner import ModelRunner


_ENABLE_METRICS_DP_ATTENTION = envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.get()


def _resolve_elastic_world_dp_size(
    dp_size: int,
    *,
    group: torch.distributed.ProcessGroup,
    local_num_tokens: int,
    local_forward_mode: int,
) -> int:
    if not world_dp_gather_enabled():
        return dp_size

    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager
    from sglang.srt.layers.dp_attention import get_attention_dp_size

    live_dp_size = get_attention_dp_size()
    effective_ep_size = ElasticEPStateManager.get_effective_ep_size()
    world_size = torch.distributed.get_world_size(group)

    if live_dp_size != effective_ep_size:
        raise RuntimeError(
            "[Elastic EP] WORLD MLP sync dp_size is out of sync: "
            f"rank={torch.distributed.get_rank(group)} "
            f"live_dp_size={live_dp_size} effective_ep_size={effective_ep_size} "
            f"world_size={world_size} server_args_dp_size={dp_size} "
            f"local_num_tokens={local_num_tokens} "
            f"local_forward_mode={local_forward_mode}"
        )
    if live_dp_size > world_size:
        raise RuntimeError(
            "[Elastic EP] WORLD MLP sync dp_size exceeds WORLD size: "
            f"rank={torch.distributed.get_rank(group)} "
            f"live_dp_size={live_dp_size} world_size={world_size} "
            f"effective_ep_size={effective_ep_size}"
        )

    return live_dp_size


# Column layout of the MLP-sync payload. This tensor crosses an all_gather, so
# an index shift here desynchronises ranks silently instead of raising -- name
# the columns rather than indexing by literal.
_SYNC_NUM_TOKENS = 0
_SYNC_NUM_TOKENS_FOR_LOGPROB = 1
_SYNC_NUM_TOKENS_DRAFT = 2
_SYNC_NUM_TOKENS_TARGET = 3
_SYNC_CAN_RUN_DECODE_CUDA_GRAPH = 4
_SYNC_IS_EXTEND_IN_BATCH = 5
_SYNC_LOCAL_CAN_RUN_TBO = 6
_SYNC_LOCAL_FORWARD_MODE = 7
_SYNC_CAN_RUN_PREFILL_CUDA_GRAPH = 8
_SYNC_NUM_COLS = 9


@dataclass
class MLPSyncBatchInfo:
    dp_size: int
    tp_size: int
    cp_size: int

    num_tokens: int
    num_tokens_for_logprob: int
    # Tokens this rank pushes through ONE draft-model forward. On a
    # mode-homogeneous step every rank derives this from `num_tokens` via
    # spec_scale_global_num_tokens, but that scaling is a single uniform
    # multiplier over the whole gathered vector: it cannot describe a step
    # where one rank prefills (its entry is already a token count) while
    # another decodes (its entry is a request count awaiting the spec width).
    # Gathering the draft-phase token count explicitly is what lets those two
    # coexist. 0 on a rank with no decode work -- including a prefilling rank,
    # whose draft slots are idle filler.
    num_tokens_draft: int
    # Tokens this rank pushes through the TARGET forward, already in token
    # units. global_num_tokens is mixed-unit -- a token count on an extending
    # rank, a request count awaiting the spec width on a decoding one -- which
    # the single uniform scale factor cannot reconcile once both appear in the
    # same step. Used in place of it (unscaled) when dp_mixed_step is set.
    num_tokens_target: int
    can_run_decode_cuda_graph: bool
    can_run_prefill_cuda_graph: bool
    is_extend_in_batch: bool
    local_can_run_tbo: bool
    local_forward_mode: int

    # some gathered elements
    has_decode_in_batch: bool = False
    dp_mixed_step: bool = False

    tp0_info_cpu: torch.Tensor = None
    global_num_tokens: list[int] = None
    global_num_tokens_draft: list[int] = None
    global_num_tokens_target: list[int] = None
    global_num_tokens_for_logprob: list[int] = None
    tbo_split_seq_index: torch.Tensor = None
    global_forward_mode: int = None
    dp_cooperation_info: Optional[DPCooperationInfo] = None

    def _get_local_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                self.num_tokens,
                self.num_tokens_for_logprob,
                self.num_tokens_draft,
                self.num_tokens_target,
                int(self.can_run_decode_cuda_graph),
                int(self.is_extend_in_batch),
                int(self.local_can_run_tbo),
                self.local_forward_mode,
                int(self.can_run_prefill_cuda_graph),
            ],
            device=device,
            dtype=dtype,
        )

    def _get_fallback_tensor(self, device, dtype=torch.int64) -> torch.Tensor:
        return torch.tensor(
            [
                0,  # num_tokens
                0,  # num_tokens_for_logprob
                0,  # num_tokens_draft
                0,  # num_tokens_target
                1,  # can_run_decode_cuda_graph
                0,  # is_extend_in_batch
                1,  # local_can_run_tbo
                ForwardMode.IDLE.value,  # local_forward_mode
                0,  # can_run_prefill_cuda_graph
            ],
            device=device,
            dtype=dtype,
        )

    def finalize_local(self):
        """Populate gather-derived metadata from the sole attention-DP rank."""
        self.tp0_info_cpu = self._get_local_tensor(device="cpu").view(1, -1)
        self.global_num_tokens = [self.num_tokens]
        self.global_num_tokens_for_logprob = [self.num_tokens_for_logprob]
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(
                self.tp0_info_cpu[:, _SYNC_LOCAL_FORWARD_MODE].tolist()
            )

    def all_gather(
        self,
        device,
        group: torch.distributed.ProcessGroup,
        use_all_reduce: bool = False,
    ):
        local_info_tensor = self._get_local_tensor(device=device)
        fallback_tensor = self._get_fallback_tensor(device=device)
        info_width = local_info_tensor.numel()
        # Inactive max_world_size slots must decode as IDLE. repeat() (not
        # expand().contiguous()) so the buffer never aliases fallback_tensor:
        # at world size 1 the expanded view is already contiguous, contiguous()
        # is a no-op, and the masked fallback writes below would then read and
        # write the same storage.
        global_info_tensor = fallback_tensor.repeat(
            self.dp_size, self.tp_size * self.cp_size, 1
        )

        if use_all_reduce:
            # Admission can expose different WORLD sizes; use fixed global slots.
            global_info_tensor.zero_()
            flat_info = global_info_tensor.view(-1, info_width)
            rank = torch.distributed.get_rank(group)
            if 0 <= rank < flat_info.shape[0]:
                flat_info[rank] = local_info_tensor
            torch.distributed.all_reduce(
                global_info_tensor,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )
            missing = flat_info.abs().sum(dim=1) == 0
            flat_info[missing] = fallback_tensor
        else:
            torch.distributed.all_gather_into_tensor(
                global_info_tensor.flatten(),
                local_info_tensor,
                group=group,
            )

        tp_info = global_info_tensor.view(
            self.dp_size * self.tp_size * self.cp_size, info_width
        )
        num_ranks_in_tp_info = tp_info.shape[0]
        if device == "cpu":
            tp_active_ranks = get_tp_group().active_ranks_cpu
        else:
            tp_active_ranks = get_tp_group().active_ranks
        if tp_active_ranks.shape[0] < num_ranks_in_tp_info:
            tp_active_ranks = torch.ones(
                num_ranks_in_tp_info,
                dtype=tp_active_ranks.dtype,
                device=tp_active_ranks.device,
            )
        tp_info[tp_active_ranks[:num_ranks_in_tp_info] == 0] = fallback_tensor

        # One D2H for every field: each `.item()` / `.tolist()` on a device
        # tensor is its own stream sync. Copy the whole tensor, not the
        # `[:, 0, :]` slice -- that slice is non-contiguous once
        # attn_tp * attn_cp > 1, adding a gather kernel inside the wait.
        tp0_info_cpu = global_info_tensor.cpu()[:, 0, :]
        self.tp0_info_cpu = tp0_info_cpu
        self.global_num_tokens = tp0_info_cpu[:, _SYNC_NUM_TOKENS].tolist()
        self.global_num_tokens_draft = tp0_info_cpu[:, _SYNC_NUM_TOKENS_DRAFT].tolist()
        self.global_num_tokens_target = tp0_info_cpu[
            :, _SYNC_NUM_TOKENS_TARGET
        ].tolist()
        self.global_num_tokens_for_logprob = tp0_info_cpu[
            :, _SYNC_NUM_TOKENS_FOR_LOGPROB
        ].tolist()
        self.can_run_decode_cuda_graph = bool(
            tp0_info_cpu[:, _SYNC_CAN_RUN_DECODE_CUDA_GRAPH].min()
        )
        self.is_extend_in_batch = bool(tp0_info_cpu[:, _SYNC_IS_EXTEND_IN_BATCH].max())
        # A step is "mixed" when at least one rank extends and at least one
        # decodes. Derived from the already-gathered forward-mode column, so
        # every rank reaches the same verdict without another collective.
        # This is what tells an extending rank it must pad its step with idle
        # draft-propose forwards to match a decoding peer's sequence.
        modes = tp0_info_cpu[:, _SYNC_LOCAL_FORWARD_MODE].tolist()
        self.has_decode_in_batch = any(m == ForwardMode.DECODE.value for m in modes)
        self.dp_mixed_step = self.is_extend_in_batch and self.has_decode_in_batch
        self.can_run_prefill_cuda_graph = bool(
            tp0_info_cpu[:, _SYNC_CAN_RUN_PREFILL_CUDA_GRAPH].min()
        )
        if _ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(
                tp0_info_cpu[:, _SYNC_LOCAL_FORWARD_MODE].tolist()
            )


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    skip_global_metadata=False,
):
    # TODO: handle the case when moe_dense_tp_size != 1
    if not require_mlp_tp_gather:
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
        batch.global_num_tokens_draft = [mlp_sync_info.num_tokens_draft]
        batch.global_num_tokens_target = [mlp_sync_info.num_tokens_target]
    else:
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
        batch.global_num_tokens_draft = mlp_sync_info.global_num_tokens_draft
        batch.global_num_tokens_target = mlp_sync_info.global_num_tokens_target
    if not skip_global_metadata:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.dp_mixed_step = mlp_sync_info.dp_mixed_step
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_decode_cuda_graph = mlp_sync_info.can_run_decode_cuda_graph
    batch.can_run_dp_prefill_cuda_graph = mlp_sync_info.can_run_prefill_cuda_graph


def should_skip_scheduler_all_gather(dp_size: int) -> bool:
    """Return whether scheduler metadata is already local and rank-invariant.

    With one attention-DP rank there is no cross-DP state to reconcile.  The
    TP schedulers consume the same broadcast request stream, so gathering the
    identical batch mode, graph eligibility, and token counts only adds a
    device collective plus host synchronization.  Preserve the environment
    override for deployments that explicitly guarantee this invariant beyond
    DP1.
    """

    return dp_size == 1 or envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get()


def _local_decode_cuda_graph_vote(
    *,
    local_batch: Optional[ScheduleBatch],
    disable_cuda_graph: bool,
) -> bool:
    """This rank's vote for the decode graph (min-reduced across dp ranks)."""
    if disable_cuda_graph:
        return False
    return (
        local_batch is None
        or local_batch.forward_mode.is_decode_or_idle()
        or local_batch.forward_mode.is_prebuilt()
    )


def _local_prefill_cuda_graph_vote(
    *,
    local_batch: Optional[ScheduleBatch],
    prefill_graph_runner,
    coordinated_prefill: bool,
    breakable_prefill: bool,
    spec_algorithm: SpeculativeAlgorithm,
    model_config,
) -> bool:
    """This rank's vote for the prefill graph (min-reduced across dp
    ranks). Extend/mixed batches vote their own replayability; a decode
    batch eligible for the decode->extend conversion votes as its 1-token-
    extend view, so the vote and the post-sync conversion always agree."""
    if local_batch is None or local_batch.forward_mode.is_idle():
        return True
    if not coordinated_prefill:
        return False

    mode = local_batch.forward_mode
    if mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
        num_tokens = local_batch.extend_num_tokens
        input_embeds = local_batch.input_embeds
        replace_embeds = local_batch.replace_embeds
        prefix_lens = local_batch.prefix_lens
        return_logprob = local_batch.return_logprob
    elif (
        mode.is_decode()
        # Conversion replays the breakable graphs only; full's fixed
        # request-slot geometry does not cover converted decode tails.
        and breakable_prefill
        # decode->extend conversion eligibility; needs the captured-graph
        # prefill runner, not the eager fallback.
        and isinstance(prefill_graph_runner, PrefillCudaGraphRunner)
        and spec_algorithm.is_none()
        and not local_batch.return_logprob
        # Grammar FSMs advance through the decode result path only.
        and not local_batch.has_grammar
        # A converted batch takes the prefill result path, which commits beam
        # requests per-req rather than through the batch decode fold; member
        # rows also have no req of their own for the reqs-aligned extend lists.
        and all(r.beam_group is None for r in local_batch.reqs)
        # Small-bucket BCG replays amplify the a2a EP logits drift (#30898)
        # into an accuracy loss.
        and get_moe_a2a_backend().is_none()
        # The converted view lacks prepare_for_extend's mamba-track fills.
        and not uses_ssm_state(model_config)
        # HiSparse decode has its own batch lifecycle and host-offloaded KV.
        and not get_memory().enable_hisparse
        and not get_exec().overlap.enable_two_batch_overlap
        and get_cp_strategy() is None
    ):
        num_tokens = local_batch.batch_size()
        input_embeds = None
        replace_embeds = None
        prefix_lens = None
        return_logprob = False
    else:
        return False

    if prefill_graph_runner is None:
        return True
    return prefill_graph_runner.can_replay_locally(
        batch_size=local_batch.batch_size(),
        num_tokens=num_tokens,
        input_embeds=input_embeds,
        replace_embeds=replace_embeds,
        prefix_lens=prefix_lens,
        is_target_verify=mode.is_target_verify(),
        capture_hidden_mode=None,
        return_logprob=return_logprob,
        lora_ineligible=prefill_graph_runner.enable_lora,
    )


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    model_runner: ModelRunner,
    dp_size: int,
    attn_tp_size: int,
    attn_cp_size: int,
    tp_group: GroupCoordinator,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
    dwdp: bool = False,
):
    # Check if other DP workers have running batches
    if (
        local_batch is None
        or local_batch.forward_mode.is_prebuilt()
        or local_batch.forward_mode.is_idle()
    ):
        num_tokens = 0
        num_tokens_for_logprob = 0
        num_tokens_draft = 0
        num_tokens_target = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
        # Same request-count units as num_tokens, so the existing uniform
        # spec scaling reproduces today's value exactly on a homogeneous
        # decode step. What it buys is the mixed case: a prefilling rank
        # reports 0 here, and 0 survives any scale factor.
        num_tokens_draft = num_tokens
        # Pre-scaled into token units so it stays comparable with an extending
        # peer's entry; the verify forward is num_draft_tokens wide per request.
        num_tokens_target = num_tokens * max(
            1, get_spec().speculative_num_draft_tokens or 1
        )
    else:
        num_tokens = local_batch.extend_num_tokens
        # An extending rank issues no drafts of its own; its draft slots are
        # idle filler that must still enter the collective.
        num_tokens_draft = 0
        # Already a token count.
        num_tokens_target = num_tokens
        num_tokens_for_logprob = sum(
            # We should have at least 1 token for sample in every case.
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                local_batch.extend_logprob_start_lens,
                local_batch.extend_lens,
            )
        )
        assert (
            local_batch.return_logprob
            or num_tokens_for_logprob == local_batch.batch_size()
        )

    can_run_decode_cuda_graph = _local_decode_cuda_graph_vote(
        local_batch=local_batch, disable_cuda_graph=disable_cuda_graph
    )
    breakable_prefill = check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE)
    coordinated_prefill = breakable_prefill or check_cuda_graph_backend(
        Phase.PREFILL, Backend.FULL
    )
    prefill_graph_runner = (
        model_runner.prefill_cuda_graph_runner if coordinated_prefill else None
    )
    can_run_prefill_cuda_graph = _local_prefill_cuda_graph_vote(
        local_batch=local_batch,
        prefill_graph_runner=prefill_graph_runner,
        coordinated_prefill=coordinated_prefill,
        breakable_prefill=breakable_prefill,
        spec_algorithm=model_runner.spec_algorithm,
        model_config=model_runner.model_config,
    )

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    tbo_preparer = TboDPAttentionPreparer()
    use_world_group = world_dp_gather_enabled()
    if use_world_group:
        from sglang.srt.distributed.parallel_state import get_world_group

        world = get_world_group()
        group = torch.distributed.group.WORLD
        device = world.device
    elif len(offload_tags) == 0 and (
        disable_overlap_schedule
        or envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)
    if use_world_group:
        dp_size = _resolve_elastic_world_dp_size(
            dp_size,
            group=group,
            local_num_tokens=num_tokens,
            local_forward_mode=local_forward_mode,
        )
    skip_all_gather = should_skip_scheduler_all_gather(dp_size)

    mlp_sync_info = MLPSyncBatchInfo(
        dp_size=dp_size,
        tp_size=attn_tp_size,
        cp_size=attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        num_tokens_draft=num_tokens_draft,
        num_tokens_target=num_tokens_target,
        can_run_decode_cuda_graph=can_run_decode_cuda_graph,
        can_run_prefill_cuda_graph=can_run_prefill_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
    )

    if dp_size == 1:
        mlp_sync_info.finalize_local()
    elif not skip_all_gather:
        mlp_sync_info.all_gather(
            device=device,
            group=group,
            use_all_reduce=use_world_group,
        )

    metadata_ready = mlp_sync_info.tp0_info_cpu is not None
    if metadata_ready:
        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info_cpu[
                    :, _SYNC_LOCAL_CAN_RUN_TBO : _SYNC_LOCAL_FORWARD_MODE + 1
                ],
            )
        )

    # Decide whether to emit idle batch
    if skip_all_gather:
        # Skip idle batch when attn-dp=1 (and always under DWDP: ranks run independently)
        need_idle_batch = not dwdp and dp_size > 1
    else:
        need_idle_batch = max(mlp_sync_info.global_num_tokens) > 0

    batch_to_gather = local_batch
    if need_idle_batch:
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()

    if batch_to_gather is not None:
        _update_gather_batch(
            batch_to_gather,
            mlp_sync_info,
            require_mlp_tp_gather,
            skip_global_metadata=not metadata_ready,
        )

    # Set on `local_batch`, not `batch_to_gather`: for PREBUILT batches the
    # scheduler's `last_batch` is the prebuilt batch, not its inner idle batch.
    if local_batch is not None and metadata_ready:
        local_batch.recv_skipper_forward_mode = (
            SchedulerRecvSkipper.derive_forward_mode(
                mlp_sync_info.tp0_info_cpu[:, _SYNC_LOCAL_FORWARD_MODE].tolist()
            )
        )

    if _ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info

    return local_batch


@dataclass(kw_only=True, slots=True, frozen=True)
class SchedulerDPAttnAdapter:
    model_runner: ModelRunner
    tp_group: GroupCoordinator
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    tree_cache: BasePrefixCache
    offload_tags: set[str]
    ps: ParallelState
    model_config: ModelConfig
    enable_overlap: bool
    spec_algorithm: SpeculativeAlgorithm
    get_require_mlp_sync: Callable[[], bool]

    def prepare_mlp_sync_batch(self, local_batch: ScheduleBatch):
        return prepare_mlp_sync_batch_raw(
            local_batch,
            model_runner=self.model_runner,
            dp_size=get_parallel().dp_size,
            attn_tp_size=self.ps.attn_tp_size,
            attn_cp_size=self.ps.attn_cp_size,
            tp_group=self.tp_group,
            get_idle_batch=self.get_idle_batch,
            disable_cuda_graph=cuda_graph_fully_disabled(),
            require_mlp_tp_gather=require_mlp_tp_gather(),
            disable_overlap_schedule=get_schedule().disable_overlap_schedule,
            offload_tags=self.offload_tags,
            dwdp=get_parallel().dwdp_size > 1,
        )

    def maybe_prepare_mlp_sync_batch(
        self,
        batch: Optional[ScheduleBatch],
        need_sync: Optional[bool] = None,
    ) -> Optional[ScheduleBatch]:
        """
        Helper to prepare MLP sync batch for DP attention.
        Should be called after get_new_batch_prefill().

        Args:
            batch: The batch to process
            need_sync: If specified, overrides self.get_require_mlp_sync() for prepare_mlp_sync_batch decision
        """
        if need_sync if need_sync is not None else self.get_require_mlp_sync():
            batch = self.prepare_mlp_sync_batch(batch)
        return batch

    def maybe_convert_decode_to_extend(
        self, batch: Optional[ScheduleBatch]
    ) -> Optional[ScheduleBatch]:
        """After the mlp-sync gather: convert an eligible decode batch to the
        extend view when a peer rank runs extend this step, so the step stays
        mode-homogeneous and every rank replays the extend graphs instead of
        all falling to eager."""
        if batch is None or not batch.forward_mode.is_decode():
            return batch
        # Global triggers from the gather. This rank's own eligibility (spec/
        # TBO/CP/logprob/replayability) is folded into the min-reduced vote:
        # if it failed, can_run_dp_prefill_cuda_graph is already False.
        if not batch.is_extend_in_batch:
            return batch
        if not batch.can_run_dp_prefill_cuda_graph:
            # The step is eager everywhere; eager decode beats eager mixed.
            return batch
        global_tokens = batch.global_num_tokens
        if (
            global_tokens is not None
            and len(global_tokens) > 1
            and min(global_tokens) == 0
        ):
            # An idle rank makes the prefill runner reject replay for every
            # rank (_has_inactive_dp_rank); converting would only trade eager
            # decode for eager mixed.
            return batch
        batch.convert_decode_to_extend()
        return batch

    def get_idle_batch(self) -> ScheduleBatch:
        idle_batch = ScheduleBatch.init_new(
            [],
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
        )
        idle_batch.prepare_for_idle()
        return idle_batch


def _debug_dp_mixed_kv(batch, tag: str) -> None:
    """Dump allocator/tree accounting around a dp-mixed conversion.

    num_used = capacity - (available + evictable); it went negative in the
    first dp-mixed run, meaning slots were counted free AND evictable at the
    same time. Gated on SGLANG_DEBUG_DP_MIXED_KV so it costs nothing normally.
    """
    if not envs.SGLANG_DEBUG_DP_MIXED_KV.get():
        return
    try:
        alloc = batch.token_to_kv_pool_allocator
        tree = batch.tree_cache
        avail = alloc.available_size()
        evict = tree.evictable_size() if hasattr(tree, "evictable_size") else -1
        rows = " ".join(
            f"{r.kv.kv_committed_len}/{r.kv.kv_allocated_len}" for r in batch.reqs[:4]
        )
        # SWA cursors: swa_evicted_seqlen only advances, and free_swa frees the
        # row range [old, new). If that range still holds spec draft-slot
        # indices from the previous speculated iteration, they get freed twice.
        swa = " ".join(
            f"{r.kv.swa_evicted_seqlen}|{r.kv.swa_evict_floor}|{r.kv.cache_protected_len}"
            for r in batch.reqs[:4]
        )
        logger.info(
            "[dp-mixed-kv] %s bs=%d avail=%d evict=%d size=%d "
            "committed/allocated: %s || swa evicted|floor|protected: %s",
            tag,
            len(batch.reqs),
            avail,
            evict,
            alloc.size,
            rows,
            swa,
        )
    except Exception as e:  # diagnostics must never take the scheduler down
        logger.info("[dp-mixed-kv] %s failed: %r", tag, e)
