//! A second, independently trainable NNUE. Its policy selects question type,
//! operand size, carry difficulty, and optional composition. The sampler
//! translates those learned choices into valid expressions.
use crate::math::{generate, Expr, Op, STAGES};
use crate::neural::{softmax, SparseNet};
use rand::Rng;
use serde::{Deserialize, Serialize};

const FEATURES:usize=15; // 10 stage bits + 5 recent-performance bits
const OUTPUTS:usize=15; // op 4, decimal width 6, hardness 3, chain 2
const GROUPS:[(usize,usize);4]=[(0,4),(4,10),(10,13),(13,15)];

#[derive(Clone,Serialize,Deserialize)]
pub struct Examiner {
    pub brain:SparseNet,
    pub trials:u64,
}
#[derive(Clone,Copy)]
pub struct ExamForce {
    pub op:Op,
    pub digits:usize,
    pub hardness:usize,
    pub chain:bool,
}
#[derive(Clone)]
pub struct Choice {
    pub op:Op,
    pub digits:usize,
    pub hardness:usize,
    pub chain:bool,
    active:[usize;2],
    selected:[usize;4],
    allowed:[Vec<usize>;4],
}
impl Choice {
    pub fn describe(&self)->String {
        format!("{} · {}자리 · 도전도 {} · {}",self.op.symbol(),self.digits,
            self.hardness+1,if self.chain {"혼합 연산"}else{"단일 연산"})
    }
}
fn select_group<R:Rng>(scores:&[f32],allowed:&[usize],rng:&mut R)->usize {
    if allowed.len()==1 {return allowed[0];}
    if rng.gen_bool(0.08) {return allowed[rng.gen_range(0..allowed.len())];}
    let logits:Vec<f32>=allowed.iter().map(|&i|scores[i]).collect();
    let probs=softmax(&logits);
    let mut v=rng.gen::<f32>();
    for (j,&p) in probs.iter().enumerate() {
        v-=p;
        if v<=0.0 {return allowed[j];}
    }
    *allowed.last().unwrap()
}
impl Examiner {
    pub fn new()->Self {Self{brain:SparseNet::new(FEATURES,56,OUTPUTS,0xE7A71),trials:0}}
    pub fn choose<R:Rng>(&self,stage:usize,recent_accuracy:f32,forced:Option<ExamForce>,rng:&mut R)->Choice {
        let index=stage.min(STAGES.len()-1);
        let profile=STAGES[index];
        let accuracy_bucket=((recent_accuracy.clamp(0.0,0.999)*5.0) as usize).min(4);
        let active=[index,10+accuracy_bucket];
        let pass=self.brain.forward(&active);
        let mut allowed:[Vec<usize>;4]=[
            profile.ops.iter().map(|op|op.index()).collect(),
            (0..profile.max_digits).map(|n|4+n).collect(),
            vec![10,11,12],
            if profile.chain {vec![13,14]} else {vec![13]},
        ];
        if let Some(f)=forced {
            allowed[0]=vec![f.op.index()];
            allowed[1]=vec![4+f.digits.clamp(1,profile.max_digits)-1];
            allowed[2]=vec![10+f.hardness.min(2)];
            allowed[3]=vec![if f.chain && profile.chain {14} else {13}];
        }
        let mut selected=[0;4];
        for i in 0..4 {
            selected[i]=select_group(&pass.scores,&allowed[i],rng);
        }
        let op=match selected[0]{0=>Op::Add,1=>Op::Sub,2=>Op::Mul,_=>Op::Div};
        Choice {
            op,digits:selected[1]-4+1,hardness:selected[2]-10,chain:selected[3]==14,
            active,selected,allowed,
        }
    }
    pub fn compose<R:Rng>(&self,stage:usize,choice:&Choice,rng:&mut R)->Expr {
        generate(stage,choice.op,choice.digits,choice.hardness,choice.chain,rng)
    }
    /// REINFORCE: improve probability of questions the student gets wrong;
    /// reduce probability of already mastered questions. Only train data updates it.
    pub fn reinforce(&mut self,choice:&Choice,reward:f32) {
        let pass=self.brain.forward(&choice.active);
        let mut delta=vec![0.0;OUTPUTS];
        for (group,&(start,end)) in GROUPS.iter().enumerate() {
            let allowed=&choice.allowed[group];
            if allowed.len()<=1 {continue;}
            let logits:Vec<f32>=allowed.iter().map(|&i|pass.scores[i]).collect();
            let probabilities=softmax(&logits);
            for (j,&absolute_index) in allowed.iter().enumerate() {
                let _ = (start,end);
                delta[absolute_index]+=reward*(probabilities[j]-if absolute_index==choice.selected[group]{1.0}else{0.0});
            }
        }
        self.brain.train(&choice.active,&pass,&delta,0.00015);
        self.trials+=1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng,SeedableRng};
    #[test]
    fn forced_exam_cannot_use_wrong_operator_or_width() {
        let mut rng=SmallRng::seed_from_u64(9);
        let e=Examiner::new();
        for _ in 0..200 {
            let chosen=e.choose(8,0.7,Some(ExamForce{op:Op::Div,digits:4,hardness:2,chain:true}),&mut rng);
            assert_eq!(chosen.op,Op::Div);
            assert_eq!(chosen.digits,4);
            assert_eq!(chosen.hardness,2);
            assert!(chosen.chain);
        }
    }
}
