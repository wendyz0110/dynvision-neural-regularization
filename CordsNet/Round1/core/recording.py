"""
Record CordsNet layer activity per 10 ms bin, through DynVision's own `record` operation.

Each CordsNet layer, at every time step: feedforward conv (previous layer, 2 ms delay)
+ recurrent conv + skip + spatial bias -> ReLU -> Euler step r = 0.8 r + 0.2 target
-> ReLU -> record. The recorded tensor (B, C, H, W) is the layer's firing rate r, the same
tensor passed on to the next layer / classifier (before any pooling).

The recorder wraps the model's single-step function: outside the 100 idle steps it asks
DynVision for the step's responses, sums them over the 5 steps of each bin and keeps the
bin mean, flattened to (B, C*H*W). Gradients flow through it when autograd is on.
The 100 idle steps are run under no_grad (see _step).
"""
import torch

N_BINS, STEPS_PER_BIN = 20, 5


class BinRecorder:
    """Attach to a CordsNet model; after model(x), `bins[layer][k]` is the (B, D) mean
    activity of `layer` in bin k (k = 0..19, bin k = stimulus steps 5k..5k+4 = 10k..10k+10 ms)."""

    def __init__(self, model, layers, grad_bins=None, steps_per_bin=STEPS_PER_BIN):
        """grad_bins: bins kept in the autograd graph: None = all bins of all layers; a list =
        the same bins for every layer; a dict {layer: list} = per layer (missing layer = none).
        The other bins are stored detached (for logging only), so they do not keep their part
        of the graph alive.
        steps_per_bin: model steps per 10 ms bin (5 at dt = 2 ms; only tests change it)."""
        self.model, self.layers, self.spb = model, list(layers), steps_per_bin
        if grad_bins is None:
            self.grad_bins = None
        elif isinstance(grad_bins, dict):
            self.grad_bins = {name: set(grad_bins.get(name, [])) for name in self.layers}
        else:
            self.grad_bins = {name: set(grad_bins) for name in self.layers}
        self.original_step = model._forward
        model._forward = self._step
        self.reset()

    def reset(self):
        self.t, self.acc, self.bins = 0, {}, {name: [None] * N_BINS for name in self.layers}

    def _step(self, x, t=None, feedforward_only=False, store_responses=False):
        m = self.model
        if m._in_idle_period:
            # DynVision runs the idle steps with autograd on and only detaches the hidden
            # states afterwards, so during training it builds (and then discards) the graph
            # of all idle steps. Running them under no_grad gives the same values and
            # roughly halves the peak training memory.
            with torch.no_grad():
                return self.original_step(x, t=t, feedforward_only=feedforward_only, store_responses=False)
        out, resp = self.original_step(x, t=t, feedforward_only=feedforward_only, store_responses=True)
        k, sub = divmod(self.t, self.spb)
        for name in self.layers:
            r = resp[name]
            if self.grad_bins is not None and k not in self.grad_bins[name]:
                r = r.detach()
            self.acc[name] = r if sub == 0 else self.acc[name] + r
            if sub == self.spb - 1:
                self.bins[name][k] = (self.acc[name] / self.spb).flatten(1)
                self.acc[name] = None
        self.t += 1
        return out, {}

    def check_complete(self, n_steps):
        assert self.t == n_steps, f"recorded {self.t} stimulus steps, expected {n_steps}"

    def detach(self):
        """Restore the model's original step function."""
        self.model._forward = self.original_step
