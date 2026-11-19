"""CPU-only soft-SLO controller. Predictions are not kernel deadlines."""
from collections import deque
import numpy as np


class AdaptivePrefill:
    def __init__(self, tpot_seconds=.1, ttft_seconds=2., minimum_tokens=128, minimum_running=8):
        self.tpot = tpot_seconds
        self.ttft = ttft_seconds
        self.minimum = minimum_tokens
        self.minimum_running = minimum_running
        self.prefill_samples = deque(maxlen=64)
        self.decode_samples = deque(maxlen=64)
        self.last_decision = {}
        self.prefill_coefficients = np.array([.3,.015,.005,.01],dtype=float)

    def observe(self, prefill, tokens, sequences, cached_tokens, seconds):
        if seconds <= 0 or not sequences:
            return
        samples = self.prefill_samples if prefill else self.decode_samples
        samples.append((tokens, sequences, cached_tokens, seconds))
        if prefill:
            # Nonnegative, regularized latency fit. Four CPU coefficients;
            # no CUDA reads or changes to model arithmetic.
            x=np.array([[t/1024,n,1,c/4096] for t,n,c,_ in samples],dtype=float)
            y=np.array([row[3] for row in samples])
            prior=np.array([.3,.015,.005,.01])
            coefficients=prior.copy()
            residual=y-x@coefficients
            for _ in range(24):
                for j in range(4):
                    column=x[:,j]
                    previous=coefficients[j]
                    value=max(0.,previous+(column@residual-1e-4*(previous-prior[j]))/(column@column+1e-4))
                    residual-=column*(value-previous)
                    coefficients[j]=value
            self.prefill_coefficients=coefficients

    def predict(self, tokens, sequences, cached_tokens=0, prefill=True):
        samples = self.prefill_samples if prefill else self.decode_samples
        if not samples:
            return (.01 + .0003*tokens + .015*sequences) if prefill else .06
        if not prefill:
            sample = min(samples, key=lambda row: abs(row[1]-sequences))
            return sample[3]*max(.5, sequences/sample[1])
        features=np.array([tokens/1024,sequences,1,cached_tokens/4096])
        return max(.002,float(features@self.prefill_coefficients))*1.1

    @staticmethod
    def shape(waiting, budget):
        tokens = sequences = cached = 0
        for seq in waiting:
            take = min(seq.num_tokens-seq.num_cached_tokens, budget-tokens)
            if take <= 0:
                break
            tokens += take
            sequences += 1
            cached += seq.num_cached_tokens
            if tokens == budget:
                break
        return tokens, sequences, cached

    def choose(self, waiting, running, maximum, now):
        """Return 0 to earn decode credit, or a positive prefill token budget.

        Credit bounds the predicted *average* TPOT for every active request.
        An oldest-waiter guard overrides credit to prevent indefinite deferral.
        Both are best-effort: a forward pass is not preemptible.
        """
        decode = self.predict(len(running), len(running), prefill=False)
        credit = min((self.tpot*seq.num_completion_tokens-(now-seq.first_token_time)-decode
                     for seq in running if seq.first_token_time is not None),default=0.)
        shape = self.shape(waiting, maximum)
        full_cost = self.predict(*shape)
        age = max(0., now-waiting[0].arrival_time)
        reason = 'credit_fit'
        budget = 0
        predicted = 0.
        if len(running)<self.minimum_running:
            budget, predicted, reason = maximum, full_cost, 'fill_decode_batch'
        elif age+decode+full_cost >= self.ttft*.8:
            budget, predicted, reason = maximum, full_cost, 'ttft_guard'
        else:
            candidates = {maximum, min(maximum,waiting[0].num_tokens-waiting[0].num_cached_tokens)}
            size = self.minimum
            while size < maximum:
                candidates.add(size)
                size *= 2
            for size in sorted(candidates):
                work = self.shape(waiting,size)
                cost = self.predict(*work)
                efficient = cost <= full_cost*work[0]/shape[0]*1.25
                if cost <= credit and efficient:
                    budget, predicted = size, cost
            if not budget:
                reason = 'earn_decode_credit'
        self.last_decision = {'reason':reason,'budget':budget,'predicted_prefill_seconds':predicted,
                              'predicted_decode_seconds':decode,'credit_seconds':credit,
                              'oldest_wait_seconds':age,'running':len(running),'waiting':len(waiting)}
        return budget
