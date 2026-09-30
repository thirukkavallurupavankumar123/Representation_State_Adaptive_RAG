# Adaptive Single-View V2 recalibration decision

**Status:** development selection only; retrieval-only; zero LLM calls; not held-out evidence.

The frozen rule selected **paragraph** retrieval at the **1,600-token operating point** with relevance percentile 0.45, patience 1, six retrieval rounds, 30 examined candidates, and a 20-chunk cap. On 63 fresh selection questions it reached recall 0.7381, answerable-only CER 0.5000, and mean actual context 1199.2 tokens. Final k was emergent, ranging from 7 to 20 (mean 15.00).

Against the same-view shallow probe, recall increased by +0.2923 (Holm p=0.00015). Against same-view k=10, recall increased by +0.0714, but v2 used 382.4 more tokens. Against same-view k=20, recall differed by -0.0053, CI [-0.0159, +0.0000], while v2 reduced context by 17.6%.

The selected point is dominated among all tested selection operating points. Frozen v1 is the sole point-estimate dominator: recall 0.7460 versus 0.7381 and actual context 1,147.6 versus 1,199.2 tokens. The paired recall difference (v2 minus v1) is -0.0079, CI [-0.0675, +0.0476], and the token difference is +51.6, CI [-2.9, +100.3]; neither is significant after correction. V1 also had a larger nominal safety maximum, so this is not evidence that v1 is superior, but it blocks a v2 replacement claim.

The required overall category is therefore **unsupported**. Scoped findings remain: v2 retrieves better than a shallow probe and k=10, and is more efficient than same-view k=20 under recall noninferiority. It does not establish held-out answer quality, universal superiority, or a benefit from representation routing.
