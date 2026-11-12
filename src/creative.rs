//! Two separately trained neural policies: problem composer and program selector.
//! Diversity pressure is measured on structural signatures, never on huge operands.
use crate::advanced::{self, Challenge, Task, SKILL_COUNT};
use crate::neural::{softmax, SparseNet};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

const COMPOSER_FEATURES: usize = 40;
const COMPOSER_OUTPUTS: usize = 48;
const STUDENT_FEATURES: usize = 3072;
const STUDENT_OUTPUTS: usize = SKILL_COUNT;
const GROUPS: [(usize,usize); 4] = [(0,30),(30,36),(36,40),(40,48)];

#[derive(Clone,Serialize,Deserialize)]
pub struct CreativeExaminer {
    pub brain: SparseNet,
    pub updates: u64,
    pub archive: NoveltyArchive,
}
#[derive(Default,Clone,Serialize,Deserialize)]
pub struct NoveltyArchive {
    recent:VecDeque<u64>,
    counts:HashMap<u64,u32>,
    pub generated:u64,
    pub fresh:u64,
    pub repeated:u64,
}
impl NoveltyArchive {
    const CAP:usize=40_000;
    pub fn novelty(&self,pattern:u64)->f32 {
        1.0/(1.0+self.counts.get(&pattern).copied().unwrap_or(0) as f32)
    }
    pub fn observe(&mut self,pattern:u64)->f32 {
        let novel=self.novelty(pattern);
        self.generated+=1;
        if novel==1.0 {self.fresh+=1;} else {self.repeated+=1;}
        self.recent.push_back(pattern);
        *self.counts.entry(pattern).or_default()+=1;
        if self.recent.len()>Self::CAP {
            if let Some(old)=self.recent.pop_front(){
                if let Some(c)=self.counts.get_mut(&old){*c-=1;if *c==0{self.counts.remove(&old);}}
            }
        }
        novel
    }
    pub fn distinct_recent(&self)->usize{self.counts.len()}
}

#[derive(Clone)]
pub struct CreativeChoice {
    pub family:usize,
    pub difficulty:usize,
    pub style:usize,
    pub mutation:usize,
    active:Vec<usize>,
    selected:[usize;4],
    allowed:[Vec<usize>;4],
}
impl CreativeChoice {
    pub fn describe(&self)->String {
        format!("{} · 강도 {} · 표현 {} · 변이 {}",advanced::skill_name(self.family),self.difficulty+1,self.style+1,self.mutation+1)
    }
}
#[derive(Clone,Copy)]
pub struct Forced {pub family:usize,pub difficulty:usize}

fn choose_one<R:Rng>(scores:&[f32],allowed:&[usize],temperature:f32,rng:&mut R)->usize {
    if allowed.len()==1{return allowed[0];}
    if rng.gen_bool(0.13){return allowed[rng.gen_range(0..allowed.len())];}
    let logits:Vec<f32>=allowed.iter().map(|&i|scores[i]/temperature).collect();
    let probabilities=softmax(&logits);
    let mut pick=rng.gen::<f32>();
    for (j,&p) in probabilities.iter().enumerate(){pick-=p;if pick<=0.0{return allowed[j]}}
    *allowed.last().unwrap()
}
impl CreativeExaminer {
    pub fn new()->Self{Self{brain:SparseNet::new(COMPOSER_FEATURES,80,COMPOSER_OUTPUTS,0x7EAE_2026),updates:0,archive:NoveltyArchive::default()}}
    pub fn choice<R:Rng>(&self,stage:usize,student_accuracy:f32,forced:Option<Forced>,rng:&mut R)->CreativeChoice {
        let s=stage.min(advanced::STAGES.len()-1);
        let stage_bucket=20+(student_accuracy.clamp(0.0,0.999)*5.0) as usize;
        let novelty_ratio=if self.archive.generated==0{1.0}else{self.archive.fresh as f32/self.archive.generated as f32};
        let novelty_bucket=25+(novelty_ratio.clamp(0.0,0.999)*5.0) as usize;
        let active=vec![s,stage_bucket.min(24),novelty_bucket.min(29),30+s.min(9)/2];
        let output=self.brain.forward(&active);
        let mut allowed:[Vec<usize>;4]=[
            advanced::STAGES[s].skills.to_vec(),
            (30..36).collect(),
            (36..40).collect(),
            (40..48).collect(),
        ];
        if let Some(f)=forced {
            allowed[0]=vec![f.family];
            allowed[1]=vec![30+f.difficulty.min(5)];
        }
        let temperature=if self.archive.generated%2500<100 {1.35}else{0.95};
        let selected=std::array::from_fn(|i|choose_one(&output.scores,&allowed[i],temperature,rng));
        CreativeChoice{family:selected[0],difficulty:selected[1]-30,style:selected[2]-36,mutation:selected[3]-40,active,selected,allowed}
    }
    pub fn propose<R:Rng>(&self,stage:usize,accuracy:f32,forced:Option<Forced>,rng:&mut R)->(CreativeChoice,Challenge) {
        // Best-of-four novelty search; balanced tests keep the forced domain.
        let mut best=None;
        for _ in 0..4 {
            let choice=self.choice(stage,accuracy,forced,rng);
            let candidate=advanced::generate(stage,choice.family,choice.difficulty,choice.style,choice.mutation,rng);
            let novelty=self.archive.novelty(candidate.pattern);
            let bonus=0.03*rng.gen::<f32>();
            let score=novelty+bonus;
            if best.as_ref().map_or(true,|(best_score,_,_)|score>*best_score){best=Some((score,choice,candidate));}
        }
        let (_,choice,candidate)=best.expect("four proposals always produce a candidate");
        (choice,candidate)
    }
    pub fn reinforce(&mut self,choice:&CreativeChoice,correct:bool,novelty:f32,valid:bool){
        let reward=if !valid{-1.0}else{
            // Encourage valid, unusual and challenging, but not simply impossible questions.
            (if correct{-0.15}else{0.45})+0.55*(novelty-0.35)
        };
        let pass=self.brain.forward(&choice.active);
        let mut gradient=vec![0.0_f32;COMPOSER_OUTPUTS];
        for (g,_) in GROUPS.iter().enumerate(){
            if choice.allowed[g].len()<2{continue;}
            let indexes=&choice.allowed[g];
            let probs=softmax(&indexes.iter().map(|&i|pass.scores[i]).collect::<Vec<_>>());
            for (j,&index) in indexes.iter().enumerate(){
                gradient[index]+=reward*(probs[j]-if index==choice.selected[g]{1.0}else{0.0});
            }
        }
        self.brain.train(&choice.active,&pass,&gradient,0.00032);
        self.updates+=1;
    }
}

#[derive(Clone,Serialize,Deserialize)]
pub struct AdvancedStudent {
    pub brain:SparseNet,
    pub lessons:u64,
}
#[derive(Clone)]
pub struct StudentResult {
    pub prediction:String,
    pub expected:String,
    pub correct:bool,
    pub completed:bool,
    pub confidence:f32,
    pub trace:Vec<String>,
    pub solved_parts:usize,
    pub total_parts:usize,
}
impl AdvancedStudent {
    pub fn new()->Self{Self{brain:SparseNet::new(STUDENT_FEATURES,64,STUDENT_OUTPUTS,0x51_u64),lessons:0}}
    pub fn classify(&self,prompt:&str)->(usize,f32){
        let features=prompt_features(prompt);
        let scores=self.brain.forward(&features).scores;
        let probabilities=softmax(&scores);
        let (index,&confidence)=probabilities.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap();
        (index,confidence)
    }
    pub fn solve(&self,challenge:&Challenge,verbose:bool)->StudentResult{
        let mut values=Vec::new();
        let mut trace=Vec::new();
        let mut solved_parts=0;
        let mut confidence=0.0;
        for (i,task) in challenge.tasks.iter().enumerate(){
            let prompt=task.prompt();
            let (program,probability)=self.classify(&prompt);
            confidence+=probability;
            let answer=advanced::execute_program(task,program);
            if answer.is_some(){solved_parts+=1;}
            let text=answer.unwrap_or_else(||"풀이 전략 미선택".to_owned());
            if verbose {
                trace.push(format!("({}) NNUE 정책: {} (확률 {:.1}%)",i+1,advanced::skill_name(program),probability*100.0));
                trace.push(format!("({}) 규칙 실행 결과: {}",i+1,text));
            }
            values.push(if challenge.tasks.len()==1{text}else{format!("({}) {}",i+1,text)});
        }
        let prediction=values.join("; ");
        let expected=challenge.expected();
        StudentResult {correct:prediction==expected && solved_parts==challenge.tasks.len(),completed:solved_parts==challenge.tasks.len(),
            confidence:confidence/challenge.tasks.len() as f32,trace,prediction,expected,solved_parts,total_parts:challenge.tasks.len()}
    }
    pub fn learn(&mut self,challenge:&Challenge){for task in &challenge.tasks{self.learn_task(task);}}
    fn learn_task(&mut self,task:&Task){
        let features=prompt_features(&task.prompt());
        let pass=self.brain.forward(&features);
        let mut gradient=softmax(&pass.scores);
        gradient[task.kind]-=1.0;
        self.brain.train(&features,&pass,&gradient,0.00075);
        self.lessons+=1;
    }
}
fn hash_feature(value:impl Hash)->usize {
    let mut h=std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut h);
    h.finish() as usize% (STUDENT_FEATURES-32)+32
}
fn prompt_features(prompt:&str)->Vec<usize>{
    let mut set=HashSet::new();
    let chars:Vec<char>=prompt.chars().collect();
    for len in [1,2,3,4]{
        if chars.len()<len{continue;}
        for chunk in chars.windows(len){
            // Numbers are normalized to D, preserving structure rather than memorizing answers.
            let token:String=chunk.iter().map(|c|if c.is_ascii_digit(){'D'}else{*c}).collect();
            set.insert(hash_feature((len,token)));
        }
    }
    set.insert((chars.len()/20).min(15));
    set.insert(16+(chars.iter().filter(|c|c.is_ascii_digit()).count()/5).min(15));
    let mut features:Vec<usize>=set.into_iter().collect();
    features.sort_unstable();features
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng,SeedableRng};
    #[test]
    fn archive_penalizes_repeating_the_same_structure(){
        let mut a=NoveltyArchive::default();
        assert_eq!(a.observe(10),1.0);
        assert!(a.observe(10)<1.0);
        assert_eq!(a.observe(20),1.0);
        assert_eq!(a.distinct_recent(),2);
    }
    #[test]
    fn examiner_and_student_are_separate_nets(){
        let mut rng=SmallRng::seed_from_u64(5);
        let examiner=CreativeExaminer::new();
        let student=AdvancedStudent::new();
        let (_,q)=examiner.propose(7,0.4,None,&mut rng);
        assert!(q.certified());
        let before=student.solve(&q,true);
        assert!(!before.trace.is_empty());
        assert_eq!(examiner.brain.outputs,COMPOSER_OUTPUTS);
        assert_eq!(student.brain.outputs,SKILL_COUNT);
    }
}
