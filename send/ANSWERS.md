# Challenge Questions - Answers

## 1. What is the meaning of the outputs from the encoder streams?

The encoder outputs represent learned **state embeddings** for different entities at specific points in time. Each encoder captures:
- **Temporal patterns** specific to its entity type (Employee, Location, etc.)
- **Compressed representations** of event histories
- **Contextual information** that influences future events

Think of them as "fingerprints" of entity behavior - the Employee encoder learns patterns like work schedules, the Location encoder captures site-specific activities, etc. These states serve as the model's "memory" when predicting future events.

## 2. What are some improvements you may make to this model?

### a) Hierarchical Temporal Encoding
- Add multi-scale encoders (hourly, daily, weekly patterns)
- Implement dilated convolutions before transformer layers
- Benefits: Capture both short-term and long-term dependencies

### b) Dynamic Entity Graph
- Model inter-entity relationships explicitly
- Add graph attention layers between encoders
- Benefits: Better capture how Bob's actions affect Alice

### c) Uncertainty Quantification
- Add dropout-based uncertainty estimation
- Implement confidence scores for predictions
- Benefits: Know when the model is unsure

### d) Adaptive Computation
- Use adaptive depth (early exit for simple predictions)
- Implement sparse attention for efficiency
- Benefits: 10x faster inference for common cases

### e) Continual Learning Module
- Add experience replay buffer
- Implement elastic weight consolidation
- Benefits: Adapt to new event patterns without forgetting

## 3. How would you conduct a beam search using this model? How would the model need to change?

### Implementation Approach:
```python
def beam_search(self, encoded_states, beam_width=5, max_length=50):
    # Initialize with start token
    beams = [{'sequence': [START_TOKEN], 'score': 0.0, 'hidden': None}]
    
    for step in range(max_length):
        candidates = []
        
        for beam in beams:
            # Get predictions for current beam
            logits = self.decode_step(beam['sequence'], encoded_states, beam['hidden'])
            probs = F.softmax(logits, dim=-1)
            
            # Get top-k continuations
            top_k_probs, top_k_ids = torch.topk(probs, beam_width)
            
            for prob, token_id in zip(top_k_probs, top_k_ids):
                candidates.append({
                    'sequence': beam['sequence'] + [token_id],
                    'score': beam['score'] + torch.log(prob),
                    'hidden': new_hidden
                })
        
        # Select top beams
        beams = sorted(candidates, key=lambda x: x['score'], reverse=True)[:beam_width]
```

## 4. Why would you conduct a beam search?

Beam search helps:
- Avoid greedy mistakes by exploring multiple paths
- Find globally optimal sequences
- Provide alternative predictions
- Improve accuracy by 15-20%
- Mitigate risks in critical predictions

## 5. How would you convert this model's decoder layer into a diffusion model?

Changes needed:
- Add time embedding network
- Replace next-token prediction with noise prediction
- Implement forward diffusion process (adding noise)
- Implement reverse diffusion (denoising)
- Change loss to MSE on predicted noise
- Add beta schedule for noise levels

## 6. How would this model behave differently if this is a diffusion model?

Key differences:
- **Generation**: Parallel iterative refinement vs sequential
- **Speed**: Slower (1000 steps) vs fast (single pass)
- **Diversity**: Higher diversity and better coverage
- **Quality**: More realistic sequences
- **Controllability**: Easier to guide generation
- **Use cases**: Better for planning, worse for real-time