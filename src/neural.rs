//! Sparse accumulator + small dense heads. All arithmetic predictions come from this network.
//! The exact arithmetic oracle lives only in the training/evaluation module.
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct SparseNet {
    pub features: usize,
    pub hidden: usize,
    pub outputs: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    m_w1: Vec<f32>, v_w1: Vec<f32>,
    m_b1: Vec<f32>, v_b1: Vec<f32>,
    m_w2: Vec<f32>, v_w2: Vec<f32>,
    m_b2: Vec<f32>, v_b2: Vec<f32>,
    step: u64,
}

pub struct Pass {
    pub hidden: Vec<f32>,
    pub scores: Vec<f32>,
}

pub fn softmax(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() { return Vec::new(); }
    let largest = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = scores.iter().map(|x| (x - largest).exp()).collect();
    let total: f32 = exps.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return vec![1.0 / scores.len() as f32; scores.len()];
    }
    for x in &mut exps { *x /= total; }
    exps
}

impl SparseNet {
    pub fn new(features: usize, hidden: usize, outputs: usize, seed: u64) -> Self {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut w1 = vec![0.0; features * hidden];
        let mut w2 = vec![0.0; hidden * outputs];
        for weight in &mut w1 { *weight = rng.gen_range(-0.24..0.24); }
        let scale = (2.0_f32 / hidden as f32).sqrt();
        for weight in &mut w2 { *weight = rng.gen_range(-scale..scale); }
        Self {
            features, hidden, outputs,
            w1, b1: vec![0.08; hidden], w2, b2: vec![0.0; outputs],
            m_w1: vec![0.0; features * hidden], v_w1: vec![0.0; features * hidden],
            m_b1: vec![0.0; hidden], v_b1: vec![0.0; hidden],
            m_w2: vec![0.0; hidden * outputs], v_w2: vec![0.0; hidden * outputs],
            m_b2: vec![0.0; outputs], v_b2: vec![0.0; outputs],
            step: 0,
        }
    }

    /// The feature accumulator is the sum of active feature-weight rows.
    /// This is the sparse first layer analogous to NNUE's accumulator.
    pub fn forward(&self, active: &[usize]) -> Pass {
        let mut h = self.b1.clone();
        for &feature in active {
            if feature >= self.features { continue; }
            let offset = feature * self.hidden;
            for (i, val) in h.iter_mut().enumerate() { *val += self.w1[offset + i]; }
        }
        for val in &mut h { *val = val.max(0.0); }
        let mut out = self.b2.clone();
        for (i, &val) in h.iter().enumerate() {
            if val == 0.0 { continue; }
            let offset = i * self.outputs;
            for (j, y) in out.iter_mut().enumerate() {
                *y += val * self.w2[offset + j];
            }
        }
        Pass { hidden: h, scores: out }
    }

    pub fn train(&mut self, active: &[usize], pass: &Pass, output_gradient: &[f32], rate: f32) {
        debug_assert_eq!(output_gradient.len(), self.outputs);
        let mut hidden_grad = vec![0.0; self.hidden];
        for i in 0..self.hidden {
            if pass.hidden[i] <= 0.0 { continue; }
            for (j, &g) in output_gradient.iter().enumerate() {
                hidden_grad[i] += g * self.w2[i * self.outputs + j];
            }
        }
        self.step += 1;
        let c1 = 1.0 - 0.9_f32.powf(self.step as f32);
        let c2 = 1.0 - 0.999_f32.powf(self.step as f32);
        for (j, &g) in output_gradient.iter().enumerate() {
            adam(&mut self.b2[j], &mut self.m_b2[j], &mut self.v_b2[j], g, rate, c1, c2);
        }
        for i in 0..self.hidden {
            let offset = i * self.outputs;
            for (j, &g) in output_gradient.iter().enumerate() {
                let idx = offset + j;
                adam(&mut self.w2[idx], &mut self.m_w2[idx], &mut self.v_w2[idx], g * pass.hidden[i], rate, c1, c2);
            }
        }
        for (i, &g) in hidden_grad.iter().enumerate() {
            adam(&mut self.b1[i], &mut self.m_b1[i], &mut self.v_b1[i], g, rate, c1, c2);
        }
        for &feature in active {
            if feature >= self.features { continue; }
            let offset = feature * self.hidden;
            for (i, &g) in hidden_grad.iter().enumerate() {
                let idx = offset + i;
                adam(&mut self.w1[idx], &mut self.m_w1[idx], &mut self.v_w1[idx], g, rate, c1, c2);
            }
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}

#[inline]
fn adam(weight: &mut f32, m: &mut f32, v: &mut f32, g: f32, rate: f32, c1: f32, c2: f32) {
    if g == 0.0 { return; }
    *m = 0.9 * *m + 0.1 * g;
    *v = 0.999 * *v + 0.001 * g * g;
    *weight -= rate * (*m / c1.max(1e-8)) / ((*v / c2.max(1e-8)).sqrt() + 1e-7);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_accumulator_and_gradient() {
        let mut net = SparseNet::new(20, 16, 3, 11);
        let before = net.forward(&[1, 3]);
        let mut g = softmax(&before.scores);
        g[2] -= 1.0;
        for _ in 0..100 {
            let p = net.forward(&[1, 3]);
            let mut grad = softmax(&p.scores);
            grad[2] -= 1.0;
            net.train(&[1, 3], &p, &grad, 0.004);
        }
        assert!(net.forward(&[1, 3]).scores[2] > before.scores[2]);
        assert_eq!(net.parameter_count(), 20*16+16+16*3+3);
    }
}
