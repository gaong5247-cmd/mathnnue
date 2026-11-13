//! Stage 30+: bounded, compositional *polynomial* adversarial curriculum.
//! A separate examiner NNUE samples expression grammar/actions and learns from errors
//! and AST novelty. A separate student NNUE selects a solution program.
//! Symbolic execution & a polynomial oracle certify answers; no arbitrary theorem proving.
use crate::creative::NoveltyArchive;
use crate::neural::{softmax, SparseNet};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{hash_map::DefaultHasher, HashSet};
use std::hash::{Hash, Hasher};

pub const FRONTIER_STAGE: usize = crate::math::STAGES.len() + crate::advanced::STAGES.len();
pub const GOAL_COUNT: usize = 7;
const NAMES: [&str; GOAL_COUNT] = [
    "함수값", "도함수", "이계도함수", "정적분",
    "계수 추출", "함수·미분 혼합", "유한차분",
];
const COMPOSER_OUTPUTS: usize = 24;
const GROUPS: [(usize, usize); 5] = [(0,7),(7,11),(11,16),(16,20),(20,24)];
const SOLVER_FEATURES: usize = 4096;
const MAX_DEGREE: usize = 8;
const COEFF_BOUND: i128 = 50_000_000;

pub fn goal_name(goal: usize) -> &'static str { NAMES[goal.min(GOAL_COUNT-1)] }

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct Weakness {
    pub attempts: u64,
    pub failures: u64,
    pub error_ema: f32,
}
impl Weakness {
    fn score(&self) -> f32 {
        // No topic is forgotten merely because it has not been attempted.
        if self.attempts < 24 { 0.52 } else { self.error_ema.max(0.04) }
    }
    fn observe(&mut self, solved: bool) {
        self.attempts += 1;
        if !solved { self.failures += 1; }
        let error = if solved { 0.0 } else { 1.0 };
        self.error_ema = if self.attempts == 1 { error } else { self.error_ema*0.97 + error*0.03 };
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Frontier {
    pub composer: SparseNet,
    pub student: SparseNet,
    pub composer_updates: u64,
    pub student_lessons: u64,
    pub archive: NoveltyArchive,
    pub weaknesses: [Weakness; GOAL_COUNT],
}
impl Default for Frontier { fn default() -> Self { Self::new() } }
impl Frontier {
    pub fn new() -> Self {
        Self {
            composer: SparseNet::new(96, 72, COMPOSER_OUTPUTS, 0xF0120_2026),
            student: SparseNet::new(SOLVER_FEATURES, 72, GOAL_COUNT, 0x5A1A_2026),
            composer_updates: 0, student_lessons: 0,
            archive: NoveltyArchive::default(),
            weaknesses: [Weakness::default(); GOAL_COUNT],
        }
    }
    pub fn weakest(&self) -> usize {
        (0..GOAL_COUNT).max_by(|&a,&b| {
            self.weaknesses[a].score().total_cmp(&self.weaknesses[b].score())
        }).unwrap_or(0)
    }
    fn context(&self) -> Vec<usize> {
        let recent = self.archive.generated;
        let fresh_ratio = if recent == 0 {1.0} else {self.archive.fresh as f32/recent as f32};
        let lessons = (self.student_lessons / 1000).min(9) as usize;
        vec![
            0, 1 + self.weakest(),
            8 + (fresh_ratio.clamp(0.0, 0.999)*8.0) as usize,
            16 + lessons,
            32 + ((self.composer_updates/1000)%16) as usize,
        ]
    }
    fn goal_bias(&self, goal: usize) -> f32 {
        // Real error memory directly changes the sampling policy.
        (self.weaknesses[goal].score() - 0.4) * 2.6
    }
    fn pick<R:Rng>(scores:&[f32], allowed:&[usize], rng:&mut R) -> usize {
        if allowed.len()==1 {return allowed[0];}
        if rng.gen_bool(0.16) {return allowed[rng.gen_range(0..allowed.len())];}
        let logits:Vec<f32>=allowed.iter().map(|&k|scores[k]/1.18).collect();
        let probabilities=softmax(&logits);
        let mut roll=rng.gen::<f32>();
        for (&candidate,p) in allowed.iter().zip(probabilities) {
            roll-=p;
            if roll<=0.0 {return candidate;}
        }
        *allowed.last().unwrap()
    }
    fn choose<R:Rng>(&self, forced_goal:Option<usize>,rng:&mut R)->Choice{
        let active=self.context();
        let mut scores=self.composer.forward(&active).scores;
        for goal in 0..GOAL_COUNT {scores[goal]+=self.goal_bias(goal);}
        let mut allowed:[Vec<usize>;5]=[
            (0..7).collect(),(7..11).collect(),(11..16).collect(),
            (16..20).collect(),(20..24).collect(),
        ];
        if let Some(goal)=forced_goal {allowed[0]=vec![goal.min(GOAL_COUNT-1)];}
        let selected=std::array::from_fn(|i|Self::pick(&scores,&allowed[i],rng));
        Choice {
            goal:selected[0], root:selected[1]-7, depth:selected[2]-11+2,
            style:selected[3]-16, mutation:selected[4]-20,
            active,allowed,selected,
        }
    }
    pub fn propose<R:Rng>(&self,forced_goal:Option<usize>,rng:&mut R)->Option<(Choice,Question)>{
        // Candidate search is structurally open: no list of fixed polynomial templates.
        let mut best:Option<(f32,Choice,Question)>=None;
        for _ in 0..28 {
            let choice=self.choose(forced_goal,rng);
            let Some(question)=Question::generate(&choice,rng) else {continue;};
            let novelty=self.archive.novelty(question.pattern);
            let vulnerability=self.weaknesses[choice.goal].score();
            let score=0.84*novelty + 0.65*vulnerability
                + 0.013*question.expr.nodes() as f32 + rng.gen::<f32>()*0.08;
            if best.as_ref().map_or(true,|(s,_,_)|score>*s){
                best=Some((score,choice,question));
            }
        }
        best.map(|(_,c,q)|(c,q))
    }
    pub fn solve(&self,q:&Question,verbose:bool)->Outcome {
        let active=prompt_features(&q.text());
        let scores=self.student.forward(&active).scores;
        let probabilities=softmax(&scores);
        let program=(0..GOAL_COUNT).max_by(|&a,&b|scores[a].total_cmp(&scores[b])).unwrap_or(0);
        let value=q.execute(program);
        let correct=value.is_some() && value==q.execute(q.goal);
        let prediction=value.unwrap_or_else(||"미해결".into());
        let expected=q.expected();
        let confidence=probabilities[program];
        let trace=if verbose {vec![
            format!("Frontier Student NNUE: {} (확률 {:.1}%)",goal_name(program),confidence*100.0),
            format!("요청 풀이: {} · 식 깊이 {} · 노드 {} · 다항식 차수 {}",
                goal_name(q.goal),q.depth(),q.expr.nodes(),q.degree()),
            format!("실행 예측: {} · 독립 다항식 채점: {}",prediction,expected),
        ]}else{vec![]};
        Outcome {prediction,expected,correct,completed:true,confidence,trace}
    }
    pub fn learn(&mut self,q:&Question,choice:&Choice,outcome:&Outcome) -> f32 {
        let novelty=self.archive.observe(q.pattern);
        self.weaknesses[q.goal].observe(outcome.correct);
        let active=prompt_features(&q.text());
        let pass=self.student.forward(&active);
        let mut grad=softmax(&pass.scores);
        grad[q.goal]-=1.0;
        self.student.train(&active,&pass,&grad,0.00072);
        self.student_lessons+=1;

        // REINFORCE on the sampled grammar actions. Avoid rewarding duplicates
        // or unsolvable garbage; only certified expressions reach this point.
        let reward=(if outcome.correct {-0.32}else{0.58})
            +0.68*(novelty-0.5) + (q.expr.nodes().min(38) as f32)*0.005;
        let mut adjusted=self.composer.forward(&choice.active).scores;
        for goal in 0..GOAL_COUNT {adjusted[goal]+=self.goal_bias(goal);}
        let mut updates=vec![0.0;COMPOSER_OUTPUTS];
        for group in 0..5 {
            if choice.allowed[group].len()<2 {continue;}
            let probs=softmax(&choice.allowed[group].iter()
                .map(|&i|adjusted[i]/1.18).collect::<Vec<_>>());
            for (i,&index) in choice.allowed[group].iter().enumerate(){
                updates[index]+=reward*(probs[i]-if index==choice.selected[group]{1.0}else{0.0});
            }
        }
        let pass=self.composer.forward(&choice.active);
        self.composer.train(&choice.active,&pass,&updates,0.00025);
        self.composer_updates+=1;
        novelty
    }
}

#[derive(Clone)]
pub struct Choice {
    pub goal:usize, pub root:usize, pub depth:usize, pub style:usize, pub mutation:usize,
    active:Vec<usize>, allowed:[Vec<usize>;5], selected:[usize;5],
}
impl Choice {
    pub fn describe(&self)->String {
        format!("{} · 문법 {} · 깊이 {} · 양식 {} · 교차변형 {}",
            goal_name(self.goal),self.root+1,self.depth,self.style+1,self.mutation+1)
    }
}

#[derive(Clone, Debug)]
pub enum Expr {
    Num(i64), X, Add(Box<Expr>,Box<Expr>), Sub(Box<Expr>,Box<Expr>),
    Mul(Box<Expr>,Box<Expr>), Pow(Box<Expr>,u8),
}
impl Expr {
    fn nodes(&self)->usize {
        match self {
            Self::Num(_) | Self::X => 1,
            Self::Pow(a,_)=>1+a.nodes(),
            Self::Add(a,b)|Self::Sub(a,b)|Self::Mul(a,b)=>1+a.nodes()+b.nodes(),
        }
    }
    fn shape<H:Hasher>(&self,hasher:&mut H){
        match self {
            Self::Num(n)=>{0u8.hash(hasher);n.signum().hash(hasher);}
            Self::X=>{1u8.hash(hasher);}
            Self::Add(a,b)=>{2u8.hash(hasher);a.shape(hasher);b.shape(hasher);}
            Self::Sub(a,b)=>{3u8.hash(hasher);a.shape(hasher);b.shape(hasher);}
            Self::Mul(a,b)=>{4u8.hash(hasher);a.shape(hasher);b.shape(hasher);}
            Self::Pow(a,p)=>{5u8.hash(hasher);p.hash(hasher);a.shape(hasher);}
        }
    }
    fn show(&self)->String {
        match self {
            Self::Num(n)=>n.to_string(),Self::X=>"x".into(),
            Self::Add(a,b)=>format!("({} + {})",a.show(),b.show()),
            Self::Sub(a,b)=>format!("({} - {})",a.show(),b.show()),
            Self::Mul(a,b)=>format!("({} × {})",a.show(),b.show()),
            Self::Pow(a,p)=>format!("({})^{}",a.show(),p),
        }
    }
    fn polynomial(&self)->Option<Vec<i128>> {
        let coeffs=match self {
            Self::Num(n)=>vec![*n as i128],
            Self::X=>vec![0,1],
            Self::Add(a,b)=>combine(&a.polynomial()?,&b.polynomial()?,1)?,
            Self::Sub(a,b)=>combine(&a.polynomial()?,&b.polynomial()?,-1)?,
            Self::Mul(a,b)=>multiply(&a.polynomial()?,&b.polynomial()?)?,
            Self::Pow(a,p)=>{
                let base=a.polynomial()?;
                let mut acc=vec![1];
                for _ in 0..*p {acc=multiply(&acc,&base)?;}
                acc
            }
        };
        bounded(coeffs)
    }
}
fn bounded(mut v:Vec<i128>)->Option<Vec<i128>> {
    while v.len()>1 && v.last()==Some(&0) {v.pop();}
    if v.len()>MAX_DEGREE+1 || v.iter().any(|x|x.abs()>COEFF_BOUND) {None}else{Some(v)}
}
fn combine(a:&[i128],b:&[i128],sign:i128)->Option<Vec<i128>>{
    let mut c=vec![0;a.len().max(b.len())];
    for (i,v) in c.iter_mut().enumerate(){
        let x=a.get(i).copied().unwrap_or(0);
        let y=b.get(i).copied().unwrap_or(0).checked_mul(sign)?;
        *v=x.checked_add(y)?;
    }
    bounded(c)
}
fn multiply(a:&[i128],b:&[i128])->Option<Vec<i128>>{
    if a.len()+b.len()>MAX_DEGREE+2 {return None;}
    let mut c=vec![0;a.len()+b.len()-1];
    for (i,&x) in a.iter().enumerate(){
        for (j,&y) in b.iter().enumerate(){
            c[i+j]=c[i+j].checked_add(x.checked_mul(y)?)?;
        }
    }
    bounded(c)
}
fn eval(p:&[i128],x:i128)->Option<i128>{
    let mut v=0_i128;
    for &a in p.iter().rev(){v=v.checked_mul(x)?.checked_add(a)?;}
    Some(v)
}
fn derivative(p:&[i128])->Vec<i128>{
    if p.len()<2 {return vec![0];}
    p.iter().enumerate().skip(1).map(|(i,&c)|c*(i as i128)).collect()
}
fn integral(p:&[i128],lo:i128,hi:i128)->Option<String>{
    // lcm(1,..,9) = 2520, so all tested polynomials have exact rational answers.
    let d=2520_i128;
    let mut sum=0_i128;
    for (i,&c) in p.iter().enumerate(){
        let power=(i+1) as u32;
        let difference=hi.checked_pow(power)?.checked_sub(lo.checked_pow(power)?)?;
        sum=sum.checked_add(c.checked_mul(d/(i as i128+1))?.checked_mul(difference)?)?;
    }
    let g=gcd(sum,d);let numerator=sum/g;let denominator=d/g;
    Some(if denominator==1 {numerator.to_string()}else{format!("{numerator}/{denominator}")})
}
fn gcd(mut a:i128,mut b:i128)->i128{
    a=a.abs();b=b.abs();
    while b!=0 {let r=a%b;a=b;b=r;}
    a.max(1)
}
fn leaf<R:Rng>(style:usize,rng:&mut R)->Expr{
    let x_weight=match style {0=>0.5,1=>0.72,2=>0.3,_=>0.58};
    if rng.gen_bool(x_weight){Expr::X}else{
        let n=if style==2 {rng.gen_range(-15..=15)}else{rng.gen_range(-8..=8)};
        Expr::Num(n)
    }
}
fn branch<R:Rng>(depth:usize,root:usize,style:usize,rng:&mut R)->Expr {
    if depth==0 || (depth>0 && rng.gen_bool(if depth>2 {0.12}else{0.28})){
        return leaf(style,rng);
    }
    let op=if rng.gen_bool(0.37){root}else{rng.gen_range(0..4)};
    let left=Box::new(branch(depth-1,root,style,rng));
    match op {
        0=>Expr::Add(left,Box::new(branch(depth-1,root,style,rng))),
        1=>Expr::Sub(left,Box::new(branch(depth-1,root,style,rng))),
        2=>Expr::Mul(left,Box::new(branch(depth.saturating_sub(2),root,style,rng))),
        _=>Expr::Pow(left,if depth>=4 && rng.gen_bool(0.2){3}else{2}),
    }
}
#[derive(Clone)]
pub struct Question {
    pub expr:Expr,
    pub goal:usize,
    pub at:i128,
    pub upper:i128,
    pub exponent:usize,
    pub pattern:u64,
}
impl Question {
    fn generate<R:Rng>(choice:&Choice,rng:&mut R)->Option<Self>{
        let mut expr=branch(choice.depth,choice.root,choice.style,rng);
        let k=rng.gen_range(1_i64..=7);
        let x=Expr::X;
        // Real transformations that change the polynomial rather than pad with x-x.
        expr=match choice.mutation {
            1=>Expr::Add(Box::new(expr),Box::new(Expr::Mul(Box::new(Expr::Num(k)),Box::new(x)))),
            2=>Expr::Mul(Box::new(expr),Box::new(Expr::Add(Box::new(x),Box::new(Expr::Num(k))))),
            3=>Expr::Sub(Box::new(expr),Box::new(Expr::Pow(Box::new(x),2))),
            _=>expr,
        };
        let p=expr.polynomial()?;
        if p.len()<2 || p.iter().skip(1).all(|&v|v==0) {return None;}
        if expr.nodes()<3 || expr.nodes()>110 {return None;}
        let at=rng.gen_range(-3_i128..=3);
        let mut upper=rng.gen_range(-3_i128..=3);
        if upper==at {upper=if at<3 {at+1}else{at-1};}
        let exponent=rng.gen_range(1..p.len());
        let mut hasher=DefaultHasher::new();
        choice.goal.hash(&mut hasher);
        choice.style.hash(&mut hasher);
        expr.shape(&mut hasher);
        let pattern=hasher.finish();
        let q=Self{expr,goal:choice.goal,at,upper,exponent,pattern};
        if q.expected_checked().is_some() && q.text().len()<720 {Some(q)}else{None}
    }
    pub fn degree(&self)->usize {self.expr.polynomial().map_or(0,|p|p.len()-1)}
    pub fn depth(&self)->usize {self.expr.nodes()}
    pub fn text(&self)->String {
        let header=format!("P(x) = {}",self.expr.show());
        let question=match self.goal {
            0=>format!("P({})를 구하시오.",self.at),
            1=>format!("P'({})를 구하시오.",self.at),
            2=>format!("P''({})를 구하시오.",self.at),
            3=>format!("∫[{}..{}] P(x) dx의 정확한 값을 구하시오.",self.at,self.upper),
            4=>format!("P(x)의 x^{} 항의 계수를 구하시오.",self.exponent),
            5=>format!("P'({}) - 2P({})를 구하시오.",self.at,self.at),
            _=>format!("P({}) - P({})를 구하시오.",self.at+1,self.at),
        };
        format!("{header}\n{question}")
    }
    fn expected_checked(&self)->Option<String>{self.execute(self.goal)}
    pub fn expected(&self)->String{self.expected_checked().unwrap_or_else(||"검증 불가".into())}
    /// Execute a selected *program*, never consult which goal was requested.
    /// The NNUE must infer the right program from the question text.
    pub fn execute(&self,program:usize)->Option<String>{
        let p=self.expr.polynomial()?;
        let answer=match program {
            0=>eval(&p,self.at)?.to_string(),
            1=>eval(&derivative(&p),self.at)?.to_string(),
            2=>eval(&derivative(&derivative(&p)),self.at)?.to_string(),
            3=>integral(&p,self.at,self.upper)?,
            4=>p.get(self.exponent).copied().unwrap_or(0).to_string(),
            5=>eval(&derivative(&p),self.at)?.checked_sub(
                2_i128.checked_mul(eval(&p,self.at)?)?)?.to_string(),
            6=>eval(&p,self.at+1)?.checked_sub(eval(&p,self.at)?)?.to_string(),
            _=>return None,
        };
        Some(answer)
    }
    pub fn verified(&self)->bool {
        let Some(p)=self.expr.polynomial() else{return false;};
        if p.len()<2 || self.expected_checked().is_none(){return false;}
        // A second evaluator traverses the tree at independent sample points.
        [-2_i128,0,2].iter().all(|&x|self.eval_tree(x)==eval(&p,x))
    }
    fn eval_tree(&self,x:i128)->Option<i128>{
        fn walk(e:&Expr,x:i128)->Option<i128>{
            match e {
                Expr::Num(n)=>Some(*n as i128),Expr::X=>Some(x),
                Expr::Add(a,b)=>walk(a,x)?.checked_add(walk(b,x)?),
                Expr::Sub(a,b)=>walk(a,x)?.checked_sub(walk(b,x)?),
                Expr::Mul(a,b)=>walk(a,x)?.checked_mul(walk(b,x)?),
                Expr::Pow(a,p)=>walk(a,x)?.checked_pow(*p as u32),
            }
        }
        walk(&self.expr,x)
    }
}
pub struct Outcome {
    pub prediction:String,pub expected:String,pub correct:bool,
    pub completed:bool,pub confidence:f32,pub trace:Vec<String>,
}
fn feature_hash<T:Hash>(v:T)->usize{
    let mut h=DefaultHasher::new();v.hash(&mut h);
    32+h.finish() as usize % (SOLVER_FEATURES-32)
}
fn prompt_features(text:&str)->Vec<usize>{
    let mut set=HashSet::new();
    let chars:Vec<char>=text.chars().collect();
    for n in [1,2,3] {
        for chunk in chars.windows(n){
            let normalized:String=chunk.iter().map(|c|if c.is_ascii_digit(){'D'}else{*c}).collect();
            set.insert(feature_hash((n,normalized)));
        }
    }
    set.insert((chars.len()/24).min(15));
    set.insert(16+(chars.iter().filter(|c|*c=='x').count()/5).min(15));
    let mut active:Vec<usize>=set.into_iter().collect();
    active.sort_unstable();active
}
#[cfg(test)]
mod tests{
    use super::*;
    use rand::{rngs::SmallRng,SeedableRng};
    #[test]
    fn diverse_verified_adversarial_grammar(){
        let mut rng=SmallRng::seed_from_u64(41);
        let frontier=Frontier::new();
        let mut structures=HashSet::new();
        for i in 0..700{
            let (choice,q)=frontier.propose(Some(i%GOAL_COUNT),&mut rng).expect("generates valid ASTs");
            assert_eq!(choice.goal,i%GOAL_COUNT);
            assert!(q.verified(),"{} -> {}",q.text(),q.expected());
            structures.insert(q.pattern);
        }
        assert!(structures.len()>240,"novel structures: {}",structures.len());
    }
    #[test]
    fn frozen_exam_cannot_mutate_weights_or_archive(){
        let mut rng=SmallRng::seed_from_u64(12);
        let frontier=Frontier::new();
        let (choice,q)=frontier.propose(Some(3),&mut rng).unwrap();
        assert_eq!(choice.goal,3);
        let before=frontier.composer_updates;
        let fresh=frontier.archive.generated;
        let _=frontier.solve(&q,false);
        assert_eq!(frontier.composer_updates,before);
        assert_eq!(frontier.archive.generated,fresh);
    }
    #[test]
    fn examiner_responds_to_weakness_and_learns(){
        let mut rng=SmallRng::seed_from_u64(16);
        let mut frontier=Frontier::new();
        let (choice,q)=frontier.propose(None,&mut rng).unwrap();
        let outcome=frontier.solve(&q,true);
        let _=frontier.learn(&q,&choice,&outcome);
        assert_eq!(frontier.student_lessons,1);
        assert_eq!(frontier.composer_updates,1);
        assert_eq!(frontier.weaknesses[q.goal].attempts,1);
        assert_eq!(frontier.archive.generated,1);
    }
}