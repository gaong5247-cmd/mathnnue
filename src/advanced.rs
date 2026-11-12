//! Verified mathematics beyond integer arithmetic.
//! This is a generative syllabus, not a claim of general theorem proving.
//! A trained NNUE chooses a *solution program*; that program applies explicit
//! symbolic rules. The answer oracle is only used for teaching and grading.
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub const SKILL_COUNT: usize = 30;
const SKILL_NAMES: [&str; SKILL_COUNT] = [
    "유리수 연산", "일차방정식", "연립방정식", "이차방정식", "삼차방정식", "사차방정식",
    "등차수열", "등비수열", "로그", "삼각함수", "극한", "미분", "정적분",
    "편미분", "행렬식", "이항계수", "확률", "중국인의 나머지 정리", "모듈러 거듭제곱",
    "최대공약수/최소공배수", "좌표기하", "부등식 최적화", "전사함수의 개수", "목걸이 색칠",
    "미분방정식", "무한등비급수", "올림피아드 부등식", "정수해의 개수", "고유값", "점화식",
];

pub fn skill_name(skill: usize) -> &'static str { SKILL_NAMES[skill.min(SKILL_COUNT - 1)] }

#[derive(Clone, Copy)]
pub struct AdvancedStage {
    pub name: &'static str,
    pub skills: &'static [usize],
    pub min_parts: usize,
    pub max_parts: usize,
}
const ALL: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29];
pub const STAGES: [AdvancedStage; 19] = [
    AdvancedStage { name: "중학 준비 · 분수/미지수", skills: &[0, 1], min_parts: 1, max_parts: 1 },
    AdvancedStage { name: "중1 · 일차방정식과 연립", skills: &[1, 2], min_parts: 1, max_parts: 1 },
    AdvancedStage { name: "중3 · 이차방정식과 수열", skills: &[3, 6], min_parts: 1, max_parts: 1 },
    AdvancedStage { name: "고1 · 삼차/사차 다항식", skills: &[3, 4, 5], min_parts: 1, max_parts: 1 },
    AdvancedStage { name: "고1 · 수열과 로그/점화식", skills: &[6, 7, 8, 29], min_parts: 1, max_parts: 1 },
    AdvancedStage { name: "고2 · 삼각함수·극한·부등식", skills: &[9, 10, 21], min_parts: 1, max_parts: 1 },
    AdvancedStage { name: "고3 · 미분·적분", skills: &[11, 12, 13], min_parts: 1, max_parts: 1 },
    AdvancedStage { name: "대학 · 선형대수", skills: &[14, 28, 2], min_parts: 1, max_parts: 1 },
    AdvancedStage { name: "대학 · 이산수학과 확률", skills: &[15, 16, 27], min_parts: 1, max_parts: 1 },
    AdvancedStage { name: "대학 · 정수론", skills: &[17, 18, 19], min_parts: 1, max_parts: 1 },
    AdvancedStage { name: "대학 · 해석·미분방정식", skills: &[12, 24, 25, 13], min_parts: 1, max_parts: 2 },
    AdvancedStage { name: "대학 · 조합/기하/군작용", skills: &[20, 22, 23], min_parts: 1, max_parts: 2 },
    AdvancedStage { name: "IMO 준비 · 대수/부등식", skills: &[3, 4, 21, 26], min_parts: 1, max_parts: 2 },
    AdvancedStage { name: "IMO 준비 · 정수론", skills: &[17, 18, 19, 27], min_parts: 1, max_parts: 2 },
    AdvancedStage { name: "IMO 준비 · 조합", skills: &[15, 22, 23, 27], min_parts: 1, max_parts: 3 },
    AdvancedStage { name: "IMO 혼합 · 검증 가능한 합성문제", skills: &[2, 3, 4, 17, 20, 21, 22, 23, 26], min_parts: 2, max_parts: 3 },
    AdvancedStage { name: "대학원 주제 · 심화 합성", skills: &[5, 11, 12, 13, 24, 25, 28], min_parts: 2, max_parts: 3 },
    AdvancedStage { name: "연구형 · 정수론/조합/대수 교차", skills: &[3, 4, 17, 18, 22, 23, 26, 27], min_parts: 2, max_parts: 4 },
    AdvancedStage { name: "무한 도전 · 전 영역 조합 (인증형)", skills: ALL, min_parts: 2, max_parts: 4 },
];

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct Task {
    pub kind: usize,
    pub p: [i64; 8],
    pub style: u8,
}
impl Task {
    pub fn title(&self) -> &'static str { skill_name(self.kind) }
    pub fn prompt(&self) -> String {
        let p = self.p;
        let core = match self.kind {
            0 => format!("{}/{} + {}/{} = ? (기약분수로)", p[0], p[1], p[2], p[3]),
            1 => format!("{}x {} = {} 일 때 x를 구하시오", p[0], signed_plus(p[1]), p[0]*p[2]+p[1]),
            2 => format!("{}x + {}y = {},  {}x + {}y = {}의 (x,y)를 구하시오", p[0],p[1],p[0]*p[4]+p[1]*p[5],p[2],p[3],p[2]*p[4]+p[3]*p[5]),
            3 | 4 | 5 => format!("{} = 0의 모든 실근을 구하시오", polynomial(&p[0..(self.kind - 1)])),
            6 => format!("등차수열 a₁={}, 공차={}일 때 a_{} = ?",p[0],p[1],p[2]),
            7 => format!("등비수열 a₁={}, 공비={}일 때 a_{} = ?",p[0],p[1],p[2]),
            8 => format!("log_{}({}^{}) = ?",p[0],p[0],p[1]),
            9 => format!("{}({}π/6)의 정확한 값을 구하시오",if p[0]==0 {"sin"}else{"cos"},p[1]),
            10 => format!("lim (x→{}) (x²-{})/(x-{}) = ?",p[0],p[0]*p[0],p[0]),
            11 => format!("f(x)={}x³ {}x² {}x + {} 일 때 f'({}) = ?",p[0],signed_plus(p[1]),signed_plus(p[2]),p[3],p[4]),
            12 => format!("정적분 ∫₀^{} ({}x² {}x) dx = ?",p[0],p[1],signed_plus(p[2])),
            13 => format!("f(x,y)={}x² {}xy {}y² 일 때 ∂f/∂x ({},{}) = ?",p[0],signed_plus(p[1]),signed_plus(p[2]),p[3],p[4]),
            14 => format!("행렬 [[{},{}],[{},{}]]의 행렬식을 구하시오",p[0],p[1],p[2],p[3]),
            15 => format!("이항계수 C({}, {})의 값을 구하시오",p[0],p[1]),
            16 => format!("동전 {}개를 던질 때 정확히 {}개 앞면이 나올 확률은?",p[0],p[1]),
            17 => format!("0≤x<{}인 정수 x 중 x≡{} (mod {}), x≡{} (mod {})을 모두 만족하는 x는?",p[0]*p[1],p[2]%p[0],p[0],p[2]%p[1],p[1]),
            18 => format!("{}^{}을 {}로 나눈 나머지는?",p[0],p[1],p[2]),
            19 => format!("두 정수 {}, {}의 최대공약수와 최소공배수는?",p[0],p[1]),
            20 => format!("직각삼각형의 두 직각변의 길이가 {}와 {}일 때 넓이는?",p[0],p[1]),
            21 => format!("x>0일 때 x + {}/x의 최솟값과 그때의 x를 구하시오",p[0]*p[0]),
            22 => format!("{}원소 집합에서 {}원소 집합으로 가는 전사함수의 개수는?",p[0],p[1]),
            23 => format!("원형으로 배치된 {}개의 구슬을 두 색으로 칠한다. 회전만 같게 볼 때 서로 다른 색칠 수는?",p[0]),
            24 => format!("y'={}y, y(0)={}를 만족하는 함수 y(x)는?",p[0],p[1]),
            25 => format!("무한급수 ∑(n=0..∞) {}({}/{})^n 의 합은?",p[0],p[1],p[2]),
            26 => format!("실수 a,b,c≥0, a+b+c={}일 때 a²+b²+c²의 최솟값은?",p[0]),
            27 => format!("x+y+z={}의 0 이상 정수해 (x,y,z)의 개수는?",p[0]),
            28 => format!("위삼각행렬 [[{},{}],[0,{}]]의 고유값은?",p[0],p[1],p[2]),
            29 => format!("f(n+1)=f(n)+{}, f(0)={} 일 때 f({}) = ?",p[0],p[1],p[2]),
            _ => "미지원 문제".to_owned(),
        };
        match self.style % 4 {
            1 => format!("[{}] {}",self.title(),core),
            2 => format!("다음 조건을 만족시키는 값을 정확히 계산하시오: {}",core),
            3 => format!("값을 구하고 식으로 확인하시오. {}",core),
            _ => core,
        }
    }
    pub fn expected(&self) -> String { // teacher/verifier only
        let p=self.p;
        match self.kind {
            0 => rational(p[0]*p[3]+p[2]*p[1],p[1]*p[3]),
            1 => p[2].to_string(),
            2 => format!("({}, {})",p[4],p[5]),
            3 | 4 | 5 => sorted_roots(&p[0..self.kind-1]),
            6 => (p[0]+(p[2]-1)*p[1]).to_string(),
            7 => (p[0]*p[1].pow((p[2]-1) as u32)).to_string(),
            8 => p[1].to_string(),
            9 => trig(p[0],p[1]).to_owned(),
            10 => (2*p[0]).to_string(),
            11 => (3*p[0]*p[4]*p[4]+2*p[1]*p[4]+p[2]).to_string(),
            12 => rational(2*p[1]*p[0].pow(3)+3*p[2]*p[0].pow(2),6),
            13 => (2*p[0]*p[3]+p[1]*p[4]).to_string(),
            14 => (p[0]*p[3]-p[1]*p[2]).to_string(),
            15 => choose(p[0] as u64,p[1] as u64).to_string(),
            16 => rational(choose(p[0] as u64,p[1] as u64) as i64,1_i64<<p[0]),
            17 => p[2].to_string(),
            18 => modpow(p[0],p[1] as u32,p[2]).to_string(),
            19 => {let g=gcd(p[0],p[1]);format!("gcd={}, lcm={}",g,p[0]/g*p[1])},
            20 => (p[0]*p[1]/2).to_string(),
            21 => format!("최솟값 {}, x={}",2*p[0],p[0]),
            22 => onto(p[0] as u64,p[1] as u64).to_string(),
            23 => necklace(p[0] as u64).to_string(),
            24 => format!("y(x)={}e^({}x)",p[1],p[0]),
            25 => rational(p[0]*p[2],p[2]-p[1]),
            26 => rational(p[0]*p[0],3),
            27 => choose((p[0]+2) as u64,2).to_string(),
            28 => sorted_roots(&[p[0],p[2]]),
            29 => (p[1]+p[0]*p[2]).to_string(),
            _ => "미지원".into(),
        }
    }
    pub fn certified(&self) -> bool {
        if self.kind>=SKILL_COUNT {return false;}
        let p=self.p;
        if self.expected().is_empty(){return false;}
        match self.kind {
            0 => p[1]>0 && p[3]>0,
            1 => p[0]!=0 && p[0]*p[2]+p[1] == p[0]*p[2]+p[1],
            2 => p[0]*p[3]!=p[1]*p[2],
            3 | 4 | 5 => p[..self.kind-1].iter().all(|x|x.abs()<=15),
            7 => p[1]>=2 && p[2]>=1 && p[2]<=7,
            9 => (1..=3).contains(&p[1]),
            12 => p[0].abs()<=100,
            16 => p[0]<=22 && p[1]<=p[0],
            17 => gcd(p[0],p[1])==1 && p[2]>=0 && p[2]<p[0]*p[1],
            18 => p[2]>1 && p[1]>=0,
            20 => p[0]%2==0,
            21 => p[0]>0,
            22 => p[0]>=p[1] && p[1]>0,
            25 => p[2]>p[1] && p[1]>0,
            _ => true,
        }
    }
}

#[derive(Clone)]
pub struct Challenge {
    pub tasks: Vec<Task>,
    pub title: String,
    pub family: usize,
    pub difficulty: usize,
    pub style: usize,
    pub mutation: usize,
    pub pattern: u64,
}
impl Challenge {
    pub fn text(&self) -> String {
        if self.tasks.len()==1 {return self.tasks[0].prompt();}
        let lines:Vec<String>=self.tasks.iter().enumerate().map(|(i,t)|format!("({}) {}",i+1,t.prompt())).collect();
        format!("{} (복합 검증 문제)\n{}",self.title,lines.join("\n"))
    }
    pub fn expected(&self) -> String {
        self.tasks.iter().enumerate().map(|(i,t)|{
            if self.tasks.len()==1 {t.expected()} else {format!("({}) {}",i+1,t.expected())}
        }).collect::<Vec<_>>().join("; ")
    }
    pub fn certified(&self) -> bool { !self.tasks.is_empty() && self.tasks.iter().all(Task::certified) }
    pub fn kinds(&self)->Vec<usize>{self.tasks.iter().map(|t|t.kind).collect()}
}

pub fn generate<R:Rng>(stage:usize,first:usize,hardness:usize,style:usize,mutation:usize,rng:&mut R)->Challenge {
    let s=STAGES[stage.min(STAGES.len()-1)];
    let count=if stage<10 {1} else {s.min_parts+(mutation % (s.max_parts-s.min_parts+1))};
    let mut tasks=Vec::with_capacity(count);
    let first=if s.skills.contains(&first){first}else{s.skills[0]};
    for i in 0..count {
        let kind=if i==0 {first} else {s.skills[(s.skills.iter().position(|&x|x==first).unwrap_or(0)+i+mutation)%s.skills.len()]};
        tasks.push(generate_task(kind,hardness+(i/2),((style+i+mutation)%4) as u8,rng));
    }
    let mut hash=DefaultHasher::new();
    stage.hash(&mut hash);hardness.hash(&mut hash);style.hash(&mut hash);mutation.hash(&mut hash);
    for t in &tasks {t.kind.hash(&mut hash);t.style.hash(&mut hash);}
    let pattern=hash.finish();
    Challenge{tasks,title:s.name.to_owned(),family:first,difficulty:hardness,style,mutation,pattern}
}
fn generate_task<R:Rng>(kind:usize,hardness:usize,style:u8,rng:&mut R)->Task {
    let scale=(4+hardness.min(5) as i64*6).min(32);
    let mut p=[0_i64;8];
    let n=|rng:&mut R|rng.gen_range(1..=scale);
    let z=|rng:&mut R|rng.gen_range(-scale..=scale);
    match kind {
        0 => {p[0]=z(rng);p[1]=n(rng);p[2]=z(rng);p[3]=n(rng);},
        1 => {p[0]=n(rng);p[1]=z(rng);p[2]=z(rng);},
        2 => {loop {p[0]=n(rng);p[1]=z(rng);p[2]=z(rng);p[3]=n(rng);if p[0]*p[3]!=p[1]*p[2]{break;}}
            p[4]=z(rng);p[5]=z(rng);},
        3 | 4 | 5 => {for v in p.iter_mut().take(kind-1) {*v=rng.gen_range(-9..=9);}
            for i in 1..(kind-1) {if p[i]==p[0] {p[i]+=(i as i64)+1;}}},
        6 => {p[0]=z(rng);p[1]=z(rng);p[2]=rng.gen_range(2..=30);},
        7 => {p[0]=rng.gen_range(1..=9);p[1]=rng.gen_range(2..=5);p[2]=rng.gen_range(2..=6);},
        8 => {p[0]=rng.gen_range(2..=9);p[1]=rng.gen_range(1..=10);},
        9 => {p[0]=rng.gen_range(0..=1);p[1]=rng.gen_range(1..=3);},
        10 => {p[0]=z(rng);},
        11 => {p[0]=z(rng);p[1]=z(rng);p[2]=z(rng);p[3]=z(rng);p[4]=z(rng);},
        12 => {p[0]=rng.gen_range(1..=9);p[1]=3*z(rng);p[2]=2*z(rng);},
        13 => {for v in p.iter_mut().take(5){*v=z(rng)}},
        14 => {for v in p.iter_mut().take(4){*v=z(rng)}},
        15 | 16 => {p[0]=rng.gen_range(3..=if kind==16{20}else{22});p[1]=rng.gen_range(1..p[0]);},
        17 => {let primes=[5,7,11,13,17,19,23];let i=rng.gen_range(0..primes.len());let j=(i+rng.gen_range(1..primes.len()))%primes.len();p[0]=primes[i];p[1]=primes[j];p[2]=rng.gen_range(0..p[0]*p[1]);},
        18 => {p[0]=rng.gen_range(2..=31);p[1]=rng.gen_range(2..=24);p[2]=rng.gen_range(5..=101);},
        19 => {p[0]=n(rng);p[1]=n(rng);},
        20 => {p[0]=2*n(rng);p[1]=n(rng);},
        21 => {p[0]=n(rng);},
        22 => {p[1]=rng.gen_range(2..=5);p[0]=rng.gen_range(p[1]..=10);},
        23 => {p[0]=rng.gen_range(4..=16);},
        24 => {p[0]=rng.gen_range(-4..=4);p[1]=n(rng);},
        25 => {p[0]=z(rng);p[2]=rng.gen_range(3..=12);p[1]=rng.gen_range(1..p[2]);},
        26 => {p[0]=3*n(rng);},
        27 => {p[0]=rng.gen_range(3..=50);},
        28 => {p[0]=z(rng);p[1]=z(rng);p[2]=z(rng);},
        29 => {p[0]=z(rng);p[1]=z(rng);p[2]=rng.gen_range(3..=80);},
        _ => (),
    }
    Task {kind,p,style}
}

fn signed_plus(v:i64)->String{if v>=0 {format!("+ {v}")}else{format!("- {}",-v)}}
fn gcd(mut a:i64,mut b:i64)->i64{a=a.abs();b=b.abs();while b!=0{let r=a%b;a=b;b=r;}a.max(1)}
fn rational(mut n:i64,mut d:i64)->String{
    if d==0{return "undefined".into();}
    if d<0 {n=-n;d=-d;}
    let g=gcd(n,d);n/=g;d/=g;
    if d==1 {n.to_string()} else {format!("{n}/{d}")}
}
fn choose(n:u64,k:u64)->u64{if k>n{return 0}let k=k.min(n-k);let mut a=1_u64;for i in 1..=k {a=a*(n-k+i)/i;}a}
fn modpow(a:i64,n:u32,m:i64)->i64{let mut x=(a%m+m)%m;let mut e=n;let mut y=1_i64;while e>0{if e&1!=0{y=y*x%m;}x=x*x%m;e>>=1;}y}
fn onto(n:u64,k:u64)->i64{let mut s=0_i64;for j in 0..=k{s+=(if (k-j)%2==0{1}else{-1})*(choose(k,j) as i64)*(j as i64).pow(n as u32);}s}
fn necklace(n:u64)->u64{(0..n).map(|k|1_u64<<gcd(n as i64,k as i64)).sum::<u64>()/n}
fn sorted_roots(roots:&[i64])->String{let mut roots=roots.to_vec();roots.sort_unstable();roots.iter().map(|x|x.to_string()).collect::<Vec<_>>().join(", ")}
fn polynomial(roots:&[i64])->String{
    // Exact multiplication of (x-r_1)...(x-r_n) makes certified polynomials.
    let mut coefficients=vec![1_i64];
    for &r in roots {
        let mut next=vec![0_i64;coefficients.len()+1];
        for (i,&a) in coefficients.iter().enumerate(){next[i]+=a;next[i+1]-=a*r;}
        coefficients=next;
    }
    let degree=roots.len();let mut terms=Vec::new();
    for (i,&c) in coefficients.iter().enumerate(){
        if c==0 {continue;}
        let pow=degree-i;
        let term=match pow{0=>c.abs().to_string(),1=>if c.abs()==1{"x".into()}else{format!("{}x",c.abs())},_=>if c.abs()==1{format!("x^{pow}")}else{format!("{}x^{pow}",c.abs())}};
        if terms.is_empty(){terms.push(format!("{}{}",if c<0{"-"}else{""},term));}
        else{terms.push(format!(" {} {}",if c<0{"-"}else{"+"},term));}
    }
    terms.concat()
}
fn trig(func:i64,num:i64)->&'static str{match (func,num){(0,1)=>"1/2",(0,2)=>"√3/2",(0,3)=>"1",(1,1)=>"√3/2",(1,2)=>"1/2",(1,3)=>"0",_=>"미지원"}}

/// Symbolic programs, deliberately separate from `Task::expected`.
/// The NNUE must select the appropriate program from the visible prompt.
/// A wrong policy receives no answer, rather than a hidden oracle fallback.
pub fn execute_program(task:&Task,program:usize)->Option<String>{
    if program!=task.kind{return None;}
    let p=task.p;
    let answer=match program {
        0=>rational(p[0]*p[3]+p[2]*p[1],p[1]*p[3]),
        1=>rational((p[0]*p[2]+p[1])-p[1],p[0]),
        2=>{let c=p[0]*p[4]+p[1]*p[5];let f=p[2]*p[4]+p[3]*p[5];let det=p[0]*p[3]-p[1]*p[2];format!("({}, {})",(c*p[3]-p[1]*f)/det,(p[0]*f-p[2]*c)/det)},
        3 | 4 | 5 => {let polynomial_roots=&p[..program-1];let coefficients={let mut v=vec![1_i64];for &r in polynomial_roots{let mut next=vec![0;v.len()+1];for(i,&c)in v.iter().enumerate(){next[i]+=c;next[i+1]-=c*r;}v=next;}v};
            // Synthetic division at guessed integer candidates (not oracle answer).
            let mut found=Vec::new();let mut coeff=coefficients;
            for _ in 0..program-1 {
                let mut root=None;
                for guess in -35..=35 {let v=coeff.iter().fold(0_i64,|acc,&c|acc*guess+c);if v==0{root=Some(guess);break;}}
                let r=root?;found.push(r);
                let mut q=vec![coeff[0]];for &c in coeff.iter().skip(1).take(coeff.len()-2){q.push(q.last().copied().unwrap()*r+c);}coeff=q;
            }
            sorted_roots(&found)
        },
        6=>(p[0]+p[1]*(p[2]-1)).to_string(),
        7=>{let mut v=p[0];for _ in 1..p[2]{v*=p[1];}v.to_string()},
        8=>{let mut v=1_i64;let mut n=0;let target=p[0].pow(p[1] as u32);while v<target{v*=p[0];n+=1;}n.to_string()},
        9=>trig(p[0],p[1]).into(),
        10=>(2*p[0]).to_string(),
        11=>{let x=p[4];(p[0]*(x*x*3)+p[1]*(2*x)+p[2]).to_string()},
        12=>{let x=p[0];rational(2*p[1]*x*x*x+3*p[2]*x*x,6)},
        13=>(2*p[0]*p[3]+p[1]*p[4]).to_string(),
        14=>(p[0]*p[3]-p[1]*p[2]).to_string(),
        15=>{let mut r=1_u64;for i in 1..=p[1] as u64{r=r*((p[0] as u64)-p[1] as u64+i)/i;}r.to_string()},
        16=>rational(choose(p[0] as u64,p[1] as u64) as i64,2_i64.pow(p[0] as u32)),
        17=>{let mut x=0;while x<p[0]*p[1]{if x%p[0]==p[2]%p[0]&&x%p[1]==p[2]%p[1]{break;}x+=1;}x.to_string()},
        18=>modpow(p[0],p[1] as u32,p[2]).to_string(),
        19=>{let mut a=p[0];let mut b=p[1];while b!=0{let c=a%b;a=b;b=c;}format!("gcd={}, lcm={}",a,p[0]/a*p[1])},
        20=>(p[0]*p[1]/2).to_string(),
        21=>format!("최솟값 {}, x={}",p[0]*2,p[0]),
        22=>onto(p[0] as u64,p[1] as u64).to_string(),
        23=>necklace(p[0] as u64).to_string(),
        24=>format!("y(x)={}e^({}x)",p[1],p[0]),
        25=>rational(p[0]*p[2],p[2]-p[1]),
        26=>rational(p[0]*p[0],3),
        27=>choose((p[0]+2) as u64,2).to_string(),
        28=>sorted_roots(&[p[0],p[2]]),
        29=>(p[1]+p[0]*p[2]).to_string(),
        _=>return None,
    };
    Some(answer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng,SeedableRng};
    #[test]
    fn all_domains_are_certified_and_have_matching_symbolic_programs(){
        let mut rng=SmallRng::seed_from_u64(88);
        for kind in 0..SKILL_COUNT {
            for _ in 0..200 {
                let task=generate_task(kind,5,2,&mut rng);
                assert!(task.certified(),"{}: {:?}",kind,task);
                assert_eq!(execute_program(&task,kind),Some(task.expected()),"{} {:?}",kind,task);
                assert_ne!(execute_program(&task,(kind+1)%SKILL_COUNT),Some(task.expected()));
            }
        }
    }
    #[test]
    fn research_stage_creates_multiple_distinct_domains(){
        let mut rng=SmallRng::seed_from_u64(23);
        let s=STAGES.len()-1;
        let q=generate(s,11,5,2,6,&mut rng);
        assert!(q.tasks.len()>=2);
        assert!(q.certified());
        assert!(q.text().contains("복합 검증 문제"));
    }
}
