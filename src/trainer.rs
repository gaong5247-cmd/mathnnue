//! Teacher/Examiner orchestration; runs on its own OS thread so egui remains responsive.
use crate::examiner::{ExamForce, Examiner};
use crate::advanced;
use crate::frontier::{Frontier, FRONTIER_STAGE, GOAL_COUNT};
use crate::creative::{AdvancedStudent, CreativeExaminer, Forced};
use crate::math::{oracle, parse, STAGES};
use crate::solver::{expected_nodes, Student};
use rand::{rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::fs::{self, File};
use std::io::Write;
use std::sync::mpsc::{Receiver, Sender, TryRecvError};
use std::time::{Duration, Instant};

pub const EXAM_SIZE:usize=100_000;
pub const TRAIN_PER_EXAM:usize=20_000;
pub const TOTAL_STAGES:usize=FRONTIER_STAGE+1;
fn stage_name(stage:usize)->&'static str {
    if stage<STAGES.len(){STAGES[stage].name}
    else if stage<FRONTIER_STAGE {advanced::STAGES[stage-STAGES.len()].name}
    else {"30단계+ · 개방형 적대적 수식 탐색"}
}
fn checkpoint_path()->std::path::PathBuf {
    let root=std::env::var_os("LOCALAPPDATA")
        .or_else(||std::env::var_os("XDG_DATA_HOME"))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(||std::env::current_dir().unwrap_or_else(|_|std::path::PathBuf::from(".")));
    root.join("MathNNUE").join("checkpoints").join("latest.json")
}

#[derive(Clone,Serialize,Deserialize,Default)]
pub struct OpRecord {
    pub total:usize,
    pub correct:usize,
    pub completed:usize,
    pub correct_steps:usize,
    pub total_steps:usize,
}
#[derive(Clone,Serialize,Deserialize)]
pub struct ExamRecord {
    pub stage:usize,
    pub ordinal:u64,
    pub questions:usize,
    pub unique:usize,
    pub correct:usize,
    pub completed:usize,
    pub correct_steps:usize,
    pub total_steps:usize,
    pub passed:bool,
    pub per_op:[OpRecord;4],
    #[serde(default)] pub advanced:bool,
    #[serde(default)] pub structural_unique:usize,
}
impl ExamRecord {
    pub fn accuracy(&self)->f32 {ratio(self.correct,self.questions)}
    pub fn completion(&self)->f32 {ratio(self.completed,self.questions)}
    pub fn step_rate(&self)->f32 {ratio(self.correct_steps,self.total_steps)}
}
fn ratio(n:usize,d:usize)->f32 {if d==0 {0.0}else{n as f32/d as f32}}

#[derive(Serialize,Deserialize)]
pub struct BrainState {
    pub version:u32,
    pub stage:usize,
    pub mastered:bool,
    pub trained:u64,
    pub trained_this_stage:usize,
    pub exams:u64,
    pub student:Student,
    pub examiner:Examiner,
    pub history:Vec<ExamRecord>,
    #[serde(default="AdvancedStudent::new")] pub advanced_student:AdvancedStudent,
    #[serde(default="CreativeExaminer::new")] pub creative_examiner:CreativeExaminer,
    #[serde(default)] pub frontier_passes:u64,
    #[serde(default)] pub frontier:Frontier,
}
impl BrainState {
    pub fn fresh()->Self {
        Self{version:4,stage:0,mastered:false,trained:0,trained_this_stage:0,
            exams:0,student:Student::new(),examiner:Examiner::new(),history:Vec::new(),
            advanced_student:AdvancedStudent::new(),creative_examiner:CreativeExaminer::new(),frontier_passes:0,frontier:Frontier::new()}
    }
    pub fn load()->Result<Self,String> {
        let data=fs::read(checkpoint_path()).or_else(|_|fs::read("checkpoints/latest.json")).map_err(|e|format!("체크포인트 없음: {e}"))?;
        let mut state:Self=serde_json::from_slice(&data).map_err(|e|format!("체크포인트 파싱 실패: {e}"))?;
        if state.version==1 && state.examiner.brain.outputs==15 {
            // Preserve the student weights when upgrading an older examiner.
            state.version=2;
            state.examiner=Examiner::new();
        }
        if state.version==2 {state.version=3;}
        if state.version==3 {
            state.version=4;
            if state.mastered && state.stage==FRONTIER_STAGE-1 {
                state.stage=FRONTIER_STAGE;state.mastered=false;
            }
        }
        if state.version!=4 || state.stage>=TOTAL_STAGES || state.student.brain.outputs!=20 || state.examiner.brain.outputs!=35
            || state.advanced_student.brain.outputs!=advanced::SKILL_COUNT || state.creative_examiner.brain.outputs!=48 {
            return Err("지원되지 않는 체크포인트 구조입니다".into());
        }
        Ok(state)
    }
    pub fn save(&self)->Result<(),String> {
        let path=checkpoint_path();
        let folder=path.parent().ok_or("체크포인트 폴더를 찾지 못함")?;
        fs::create_dir_all(folder).map_err(|e|e.to_string())?;
        let data=serde_json::to_vec(self).map_err(|e|e.to_string())?;
        let tmp=folder.join("latest.json.tmp");
        let mut file=File::create(&tmp).map_err(|e|e.to_string())?;
        file.write_all(&data).map_err(|e|e.to_string())?;
        file.sync_all().map_err(|e|e.to_string())?;
        drop(file);
        if fs::rename(&tmp,&path).is_err() {
            if path.exists() {fs::remove_file(&path).map_err(|e|e.to_string())?;}
            fs::rename(&tmp,&path).map_err(|e|e.to_string())?;
        }
        Ok(())
    }
}

#[derive(Clone,Default)]
pub struct Snapshot {
    pub stage:usize,
    pub stage_name:String,
    pub running:bool,
    pub mastered:bool,
    pub in_exam:bool,
    pub exam_progress:usize,
    pub trained:u64,
    pub trained_this_stage:usize,
    pub micro_steps:u64,
    pub generation_trials:u64,
    pub current_question:String,
    pub current_guess:String,
    pub expected:String,
    pub neural_policy:String,
    pub trace:Vec<String>,
    pub recent_accuracy:f32,
    pub recent_completion:f32,
    pub last_exam:Option<ExamRecord>,
    pub history:Vec<ExamRecord>,
    pub student_params:usize,
    pub examiner_params:usize,
    pub message:String,
    pub novelty_recent:usize,
    pub novelty_fresh:u64,
    pub novelty_total:u64,
    pub creative_params:usize,
    pub advanced_params:usize,
    pub frontier_passes:u64,
    pub frontier_lessons:u64,
    pub frontier_new:u64,
    pub frontier_repeat:u64,
    pub frontier_weakness:String,
}
#[derive(Debug)]
pub enum Command { Start, Pause, Exam, Save, Ask(String), Generate(usize), Preview(usize,usize), PreviewFrontier(usize), Shutdown }
pub enum Event {
    Snapshot(Snapshot),
    Notice(String),
    Answer {question:String,prediction:String,expected:String,confidence:f32,trace:Vec<String>},
    Quiz(Vec<(String,String)>),
}

pub struct Worker {
    state:BrainState,
    rng:SmallRng,
    tx:Sender<Event>,
    rx:Receiver<Command>,
    running:bool,
    exam_requested:bool,
    in_exam:bool,
    exam_paused:bool,
    exam_progress:usize,
    recent:VecDeque<(bool,bool)>,
    recent_correct:usize,
    recent_completed:usize,
    current_question:String,
    current_guess:String,
    expected:String,
    neural_policy:String,
    trace:Vec<String>,
    message:String,
    last_emit:Instant,
    exiting:bool,
}
impl Worker {
    pub fn new(tx:Sender<Event>,rx:Receiver<Command>)->Self {
        let loaded=BrainState::load();
        let state=loaded.unwrap_or_else(|_|BrainState::fresh());
        let message=if state.trained>0 {format!("체크포인트 복원 완료 · 누적 {}문항",state.trained)}
            else {"처음 시작합니다. 학습 시작을 누르세요".into()};
        Self {state,rng:SmallRng::seed_from_u64(0x20261112),tx,rx,
            running:false,exam_requested:false,in_exam:false,exam_paused:false,exam_progress:0,
            recent:VecDeque::new(),recent_correct:0,recent_completed:0,
            current_question:String::new(),current_guess:String::new(),expected:String::new(),
            neural_policy:String::new(),trace:Vec::new(),message,last_emit:Instant::now(),exiting:false}
    }
    fn recent_accuracy(&self)->f32 {ratio(self.recent_correct,self.recent.len())}
    fn recent_completion(&self)->f32 {ratio(self.recent_completed,self.recent.len())}
    fn add_recent(&mut self,correct:bool,complete:bool) {
        self.recent.push_back((correct,complete));
        self.recent_correct+=usize::from(correct);
        self.recent_completed+=usize::from(complete);
        if self.recent.len()>1000 {
            if let Some((a,b))=self.recent.pop_front() {
                self.recent_correct-=usize::from(a);
                self.recent_completed-=usize::from(b);
            }
        }
    }
    fn snapshot(&self)->Snapshot {
        Snapshot {
            stage:self.state.stage,stage_name:stage_name(self.state.stage).into(),
            running:self.running,mastered:self.state.mastered,in_exam:self.in_exam,
            exam_progress:self.exam_progress,trained:self.state.trained,
            trained_this_stage:self.state.trained_this_stage,
            micro_steps:self.state.student.steps_trained,
            generation_trials:self.state.examiner.trials+self.state.creative_examiner.updates+self.state.frontier.composer_updates,
            current_question:self.current_question.clone(),current_guess:self.current_guess.clone(),
            expected:self.expected.clone(),neural_policy:self.neural_policy.clone(),trace:self.trace.clone(),
            recent_accuracy:self.recent_accuracy(),recent_completion:self.recent_completion(),
            last_exam:self.state.history.last().cloned(),history:self.state.history.clone(),
            student_params:self.state.student.brain.parameter_count(),
            examiner_params:self.state.examiner.brain.parameter_count(),
            message:self.message.clone(),
            novelty_recent:self.state.creative_examiner.archive.distinct_recent(),
            novelty_fresh:self.state.creative_examiner.archive.fresh,
            novelty_total:self.state.creative_examiner.archive.generated,
            creative_params:self.state.creative_examiner.brain.parameter_count(),
            advanced_params:self.state.advanced_student.brain.parameter_count(),
            frontier_passes:self.state.frontier_passes,
            frontier_lessons:self.state.frontier.student_lessons,
            frontier_new:self.state.frontier.archive.fresh,
            frontier_repeat:self.state.frontier.archive.repeated,
            frontier_weakness:format!("{} · 오답 EMA {:.0}%",crate::frontier::goal_name(self.state.frontier.weakest()),
                100.0*self.state.frontier.weaknesses[self.state.frontier.weakest()].error_ema),
        }
    }
    fn send_snapshot(&mut self,force:bool) {
        if force || self.last_emit.elapsed()>Duration::from_millis(180) {
            let _=self.tx.send(Event::Snapshot(self.snapshot()));
            self.last_emit=Instant::now();
        }
    }
    fn notice(&mut self,msg:impl Into<String>) {
        let msg=msg.into();self.message=msg.clone();let _=self.tx.send(Event::Notice(msg));
    }
    fn command(&mut self,cmd:Command) {
        match cmd {
            Command::Start => {self.running=true;self.exam_paused=false;self.notice("학습을 시작했습니다");self.send_snapshot(true);},
            Command::Pause => {self.running=false;if self.in_exam {self.exam_paused=true;}self.notice("일시정지됨 (체크포인트는 유지됩니다)");self.send_snapshot(true);},
            Command::Exam => {if !self.in_exam {self.exam_requested=true;self.notice("10만 문항 검증 시험을 예약했습니다");}},
            Command::Save => {let result=self.state.save();self.notice(match result {Ok(_)=>"체크포인트 저장 완료".into(),Err(e)=>format!("저장 오류: {e}")});},
            Command::Ask(question) => self.ask(question),
            Command::Generate(n) => self.quiz(n.clamp(1,40)),
            Command::Preview(stage,n) => self.preview(stage,n.clamp(1,40)),
            Command::PreviewFrontier(n) => self.preview_frontier(n.clamp(1,40)),
            Command::Shutdown => {self.exiting=true;let _=self.state.save();},
        }
    }
    fn ask(&mut self,question:String) {
        let (prediction,expected,confidence,trace)=match parse(&question) {
            Ok(expr)=>{
                let solution=self.state.student.solve(&expr,true);
                let expected=oracle(&expr).map(|n|n.to_string()).unwrap_or_else(|e|format!("검증 불가: {e}"));
                let prediction=solution.value.map(|n|n.to_string())
                    .unwrap_or_else(||format!("풀이 실패: {}",solution.error.unwrap_or_default()));
                (prediction,expected,solution.confidence,solution.trace)
            }
            Err(arithmetic_error)=>match crate::input::parse_custom(&question) {
                Ok(challenge)=>{
                    let result=self.state.advanced_student.solve(&challenge,true);
                    (result.prediction,result.expected,result.confidence,result.trace)
                }
                Err(advanced_error)=>(format!("미지원 입력 · 산술: {arithmetic_error} / 대수: {advanced_error}"),"—".into(),0.0,Vec::new()),
            },
        };
        let _=self.tx.send(Event::Answer{question,prediction,expected,confidence,trace});
    }
    fn quiz(&mut self,count:usize) {
        if self.state.stage>=FRONTIER_STAGE{self.preview_frontier(count);return;}
        if self.state.stage>=STAGES.len(){self.preview(self.state.stage-STAGES.len(),count);return;}
        let mut questions=Vec::with_capacity(count);
        for _ in 0..count {
            let c=self.state.examiner.choose(self.state.stage,self.recent_accuracy(),None,&mut self.rng);
            let q=self.state.examiner.compose(self.state.stage,&c,&mut self.rng);
            let answer=oracle(&q).map(|v|v.to_string()).unwrap_or_else(|_|"검증 불가".into());
            questions.push((q.to_string(),answer));
        }
        let _=self.tx.send(Event::Quiz(questions));
    }
    fn preview(&mut self,advanced_stage:usize,count:usize) {
        let mut questions=Vec::with_capacity(count);
        for _ in 0..count {
            let (_,challenge)=self.state.creative_examiner.propose(advanced_stage,
                self.recent_accuracy(),None,&mut self.rng);
            if challenge.certified(){questions.push((challenge.text(),challenge.expected()));}
        }
        let _=self.tx.send(Event::Quiz(questions));
        self.notice(format!("{} 창작형 출제 미리보기 · 학습 단계는 변경되지 않음",
            advanced::STAGES[advanced_stage.min(advanced::STAGES.len()-1)].name));
    }
    fn preview_frontier(&mut self,count:usize) {
        let mut questions=Vec::with_capacity(count);
        for i in 0..count {
            if let Some((_,q))=self.state.frontier.propose(Some(i%GOAL_COUNT),&mut self.rng){
                if q.verified() {questions.push((q.text(),q.expected()));}
            }
        }
        let _=self.tx.send(Event::Quiz(questions));
        self.notice("30단계+ 적대적 수식 생성 미리보기 · 학습 단계 유지");
    }
    fn train_frontier(&mut self) {
        // Retain prior high-level topics. Most stage-30 lessons use novel ASTs.
        if self.state.trained%7==0 {self.train_advanced();return;}
        let Some((choice,q))=self.state.frontier.propose(None,&mut self.rng) else {return;};
        if !q.verified(){return;}
        let verbose=self.state.trained%83==0;
        let prediction=self.state.frontier.solve(&q,verbose);
        let novelty=self.state.frontier.learn(&q,&choice,&prediction);
        self.add_recent(prediction.correct,prediction.completed);
        self.state.trained+=1;
        self.state.trained_this_stage+=1;
        if verbose {
            self.current_question=q.text();
            self.current_guess=prediction.prediction;
            self.expected=prediction.expected;
            self.neural_policy=format!("새 AST · {} · 참신도 {:.2} · 구조 #{:x}",
                choice.describe(),novelty,q.pattern);
            self.trace=prediction.trace;
        }
        if self.state.trained_this_stage>=TRAIN_PER_EXAM {self.exam_requested=true;}
        if self.state.trained%5000==0 {let _=self.state.save();}
        self.send_snapshot(false);
    }
    fn train_advanced(&mut self){
        let stage=self.state.stage-STAGES.len();
        let (choice,question)=self.state.creative_examiner.propose(stage,
            self.recent_accuracy(),None,&mut self.rng);
        let valid=question.certified();
        if !valid {self.state.creative_examiner.reinforce(&choice,false,0.0,false);return;}
        let verbose=self.state.trained%97==0;
        let solution=self.state.advanced_student.solve(&question,verbose);
        let novel=self.state.creative_examiner.archive.observe(question.pattern);
        self.state.creative_examiner.reinforce(&choice,solution.correct,novel,valid);
        self.state.advanced_student.learn(&question);
        self.add_recent(solution.correct,solution.completed);
        self.state.trained+=1;
        self.state.trained_this_stage+=1;
        if verbose {
            self.current_question=question.text();
            self.current_guess=solution.prediction;
            self.expected=solution.expected;
            self.neural_policy=format!("{} · 구조 참신성 {:.2} · 구조 #{}",choice.describe(),novel,question.pattern);
            self.trace=solution.trace;
        }
        if self.state.trained_this_stage>=TRAIN_PER_EXAM {self.exam_requested=true;}
        if self.state.trained%5000==0 {let _=self.state.save();}
        self.send_snapshot(false);
    }
    fn train_one(&mut self) {
        if self.state.stage>=FRONTIER_STAGE{self.train_frontier();return;}
        if self.state.stage>=STAGES.len(){self.train_advanced();return;}
        let choice=self.state.examiner.choose(self.state.stage,self.recent_accuracy(),None,&mut self.rng);
        let expr=self.state.examiner.compose(self.state.stage,&choice,&mut self.rng);
        let target=match oracle(&expr) {Ok(n)=>n,Err(_)=>return};
        // Student predicts first, before the teacher supplies the target.
        let verbose=self.state.trained%257==0;
        let solution=self.state.student.solve(&expr,verbose);
        let correct=solution.value==Some(target) && solution.complete();
        let complete=solution.complete();
        let reward=if correct {-0.28}else{1.0};
        self.state.examiner.reinforce(&choice,reward);
        self.state.student.learn_problem(&expr);
        // Mixed digit drills improve transfer between unseen multi-digit problems.
        self.state.student.drill(&mut self.rng);
        self.add_recent(correct,complete);
        self.state.trained+=1;
        self.state.trained_this_stage+=1;
        if verbose {
            self.current_question=expr.to_string();
            self.current_guess=solution.value.map(|v|v.to_string()).unwrap_or_else(||"미완료".into());
            self.expected=target.to_string();
            self.neural_policy=choice.describe();
            self.trace=solution.trace;
        }
        if self.state.trained_this_stage>=TRAIN_PER_EXAM {self.exam_requested=true;}
        self.send_snapshot(false);
    }
    fn handle_exam_commands(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(c)=>self.command(c),
                Err(TryRecvError::Empty)=>break,
                Err(TryRecvError::Disconnected)=>{self.exiting=true;break;},
            }
        }
        while self.exam_paused && !self.exiting && self.in_exam {
            match self.rx.recv_timeout(Duration::from_millis(120)) {
                Ok(command)=>self.command(command),
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected)=>{self.exiting=true;break;},
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)=>self.send_snapshot(false),
            }
        }
    }
    fn examination(&mut self,n:usize) {
        let stage=self.state.stage;
        if stage>=FRONTIER_STAGE{self.examination_frontier(n);return;}
        if stage>=STAGES.len(){self.examination_advanced(n);return;}
        self.in_exam=true;
        self.exam_paused=false;
        self.exam_progress=0;
        self.exam_requested=false;
        self.notice(format!("제{}회 시험 · {}단계 · {}문항 (NNUE 출제, 감독 채점)",
            self.state.exams+1,stage+1,n));
        self.send_snapshot(true);
        let mut record=ExamRecord {
            stage,ordinal:self.state.exams+1,questions:n,unique:0,correct:0,completed:0,
            correct_steps:0,total_steps:0,passed:false,per_op:std::array::from_fn(|_|OpRecord::default()),
            advanced:false,structural_unique:0,
        };
        let mut unique=HashSet::with_capacity(n.min(100_000));
        let mut exam_rng=SmallRng::seed_from_u64(0x5eed_0000+record.ordinal*1000+stage as u64);
        let profile=STAGES[stage];
        for i in 0..n {
            if i%256==0 {
                self.exam_progress=i;
                self.send_snapshot(false);
                self.handle_exam_commands();
                if self.exiting {break;}
            }
            let op=profile.ops[i % profile.ops.len()];
            let width=if stage==3 && op==crate::math::Op::Mul {1}
                else {(i / profile.ops.len()) % profile.max_digits + 1};
            let hardness=(i / (profile.ops.len()*profile.max_digits)) % 3;
            let choice=self.state.examiner.choose(stage,self.recent_accuracy(),
                Some(ExamForce{op,digits:width,hardness,
                    chain:profile.chain && (i / (profile.ops.len()*profile.max_digits*3))%2==1}),&mut exam_rng);
            let expr=self.state.examiner.compose(stage,&choice,&mut exam_rng);
            let target=match oracle(&expr) {Ok(value)=>value,Err(_)=>continue};
            let prediction=self.state.student.solve(&expr,false);
            let exact=prediction.complete() && prediction.value==Some(target);
            let step_expected={let mut v=Vec::new();expected_nodes(&expr,&mut v);v};
            let total_steps=step_expected.len();
            let step_correct=step_expected.iter().zip(prediction.nodes.iter())
                .filter(|(expected,got)|expected.is_some() && expected==got).count();
            let bin=&mut record.per_op[op.index()];
            bin.total+=1;bin.correct+=usize::from(exact);bin.completed+=usize::from(prediction.complete());
            bin.correct_steps+=step_correct;bin.total_steps+=total_steps;
            record.correct+=usize::from(exact);
            record.completed+=usize::from(prediction.complete());
            record.correct_steps+=step_correct;record.total_steps+=total_steps;
            unique.insert(expr.to_string());
            if i%1999==0 {
                let detailed=self.state.student.solve(&expr,true);
                self.current_question=expr.to_string();
                self.current_guess=prediction.value.map(|v|v.to_string()).unwrap_or_else(||"미완료".into());
                self.expected=target.to_string();
                self.neural_policy=format!("시험 문제 · {}",choice.describe());
                self.trace=detailed.trace;
            }
        }
        self.exam_progress=n;
        self.in_exam=false;
        if self.exiting {return;}
        record.unique=unique.len();
        record.passed=n==EXAM_SIZE && record.accuracy()>=0.90 && record.step_rate()>=0.90 && record.completion()>=0.90;
        self.state.exams+=1;
        self.state.history.push(record.clone());
        if self.state.history.len()>60 {self.state.history.remove(0);}
        self.state.trained_this_stage=0;
        let msg=format!("시험 완료 · 정답 {:.2}% · 풀이 단계 {:.2}% · 완주 {:.2}% · 고유문항 {}/{} · {}",
            record.accuracy()*100.0,record.step_rate()*100.0,record.completion()*100.0,
            record.unique,n,if record.passed {"통과"}else{"재학습"});
        self.notice(msg);
        if record.passed {
            if stage+1<TOTAL_STAGES {
                self.state.stage+=1;
                self.recent.clear();self.recent_correct=0;self.recent_completed=0;
                self.notice(format!("승급! 다음 단계: {}",stage_name(self.state.stage)));
            } else {
                self.state.mastered=true;
                self.state.frontier_passes+=1;
                self.notice("최종 단계 인증: 무한 도전/추가 출제 학습을 계속합니다".to_owned());
            }
        }
        if let Err(e)=self.state.save() {self.notice(format!("자동 저장 오류: {e}"));}
        self.send_snapshot(true);
    }
    fn examination_advanced(&mut self,n:usize){
        let stage=self.state.stage;
        let level=stage-STAGES.len();
        self.in_exam=true;
        self.exam_paused=false;
        self.exam_requested=false;
        self.exam_progress=0;
        self.notice(format!("제{}회 고급 시험 · {} · {}문항 (훈련과 분리된 고정 균형 검증)",
            self.state.exams+1,advanced::STAGES[level].name,n));
        self.send_snapshot(true);
        let mut record=ExamRecord{
            stage,ordinal:self.state.exams+1,questions:n,unique:0,correct:0,completed:0,
            correct_steps:0,total_steps:0,passed:false,per_op:std::array::from_fn(|_|OpRecord::default()),
            advanced:true,structural_unique:0,
        };
        let mut exact=HashSet::with_capacity(n.min(100_000));
        let mut structures=HashSet::new();
        let mut rng=SmallRng::seed_from_u64(0xADD0_2026+record.ordinal*1009+stage as u64);
        let allowed=advanced::STAGES[level].skills;
        for i in 0..n {
            if i%256==0 {
                self.exam_progress=i;
                self.send_snapshot(false);
                self.handle_exam_commands();
                if self.exiting {break;}
            }
            let family=allowed[i%allowed.len()];
            let difficulty=(i/allowed.len())%6;
            let (choice,q)=self.state.creative_examiner.propose(level,self.recent_accuracy(),
                Some(Forced{family,difficulty}),&mut rng);
            if !q.certified(){continue;}
            let solution=self.state.advanced_student.solve(&q,false);
            let group=difficulty%4;
            let stat=&mut record.per_op[group];
            stat.total+=1;
            stat.correct+=usize::from(solution.correct);
            stat.completed+=usize::from(solution.completed);
            stat.correct_steps+=solution.solved_parts;
            stat.total_steps+=solution.total_parts;
            record.correct+=usize::from(solution.correct);
            record.completed+=usize::from(solution.completed);
            record.correct_steps+=solution.solved_parts;
            record.total_steps+=solution.total_parts;
            exact.insert(q.text());
            structures.insert(q.pattern);
            if i%1999==0 {
                self.current_question=q.text();
                self.current_guess=solution.prediction;
                self.expected=solution.expected;
                self.neural_policy=format!("분리 검증 · {}",choice.describe());
                self.trace=self.state.advanced_student.solve(&q,true).trace;
            }
        }
        if self.exiting {self.in_exam=false;return;}
        self.exam_progress=n;
        self.in_exam=false;
        record.unique=exact.len();
        record.structural_unique=structures.len();
        record.passed=n==EXAM_SIZE&&record.accuracy()>=0.90&&record.step_rate()>=0.90&&record.completion()>=0.90;
        self.state.exams+=1;
        self.state.history.push(record.clone());
        if self.state.history.len()>60{self.state.history.remove(0);}
        self.state.trained_this_stage=0;
        self.notice(format!("고급 시험 · 정답 {:.2}% / 풀이 {:.2}% / 완주 {:.2}% · 구조 {}개 · {}",
            record.accuracy()*100.0,record.step_rate()*100.0,record.completion()*100.0,
            record.structural_unique,if record.passed{"승급"}else{"재학습"}));
        if record.passed {
            self.recent.clear();self.recent_correct=0;self.recent_completed=0;
            if stage+1<TOTAL_STAGES {
                self.state.stage+=1;
                self.notice(format!("새 주제 해금: {}",stage_name(self.state.stage)));
            } else {
                self.state.mastered=true;
                self.state.frontier_passes+=1;
                self.notice("인증형 최종 템플릿 통과 · 새 수학 이론 증명과는 별개입니다. 창작 학습은 계속됩니다.");
            }
        }
        if let Err(e)=self.state.save(){self.notice(format!("자동 저장 실패: {e}"));}
        self.send_snapshot(true);
    }
    fn examination_frontier(&mut self,n:usize) {
        self.in_exam=true;
        self.exam_paused=false;
        self.exam_requested=false;
        self.exam_progress=0;
        self.notice(format!("30단계+ 적대적 시험 {}문항 · 출제/풀이 신경망 가중치 고정",n));
        self.send_snapshot(true);
        let mut record=ExamRecord{
            stage:self.state.stage,ordinal:self.state.exams+1,questions:n,
            unique:0,correct:0,completed:0,correct_steps:0,total_steps:0,
            passed:false,per_op:std::array::from_fn(|_|OpRecord::default()),
            advanced:true,structural_unique:0,
        };
        let mut exact=HashSet::with_capacity(n.min(100_000));
        let mut structures=HashSet::new();
        let mut exam_rng=SmallRng::seed_from_u64(0xFEE1_2026+record.ordinal*11_113);
        for i in 0..n {
            if i%256==0 {
                self.exam_progress=i;
                self.send_snapshot(false);
                self.handle_exam_commands();
                if self.exiting {break;}
            }
            let Some((choice,q))=self.state.frontier.propose(Some(i%GOAL_COUNT),&mut exam_rng) else {continue;};
            if !q.verified() {continue;}
            let solution=self.state.frontier.solve(&q,false);
            let bin=&mut record.per_op[i%4];
            bin.total+=1;bin.correct+=usize::from(solution.correct);
            bin.completed+=usize::from(solution.completed);
            bin.correct_steps+=usize::from(solution.correct);bin.total_steps+=1;
            record.correct+=usize::from(solution.correct);
            record.completed+=usize::from(solution.completed);
            record.correct_steps+=usize::from(solution.correct);record.total_steps+=1;
            exact.insert(q.text());
            structures.insert(q.pattern);
            if i%1999==0 {
                self.current_question=q.text();
                self.current_guess=solution.prediction;
                self.expected=solution.expected;
                self.neural_policy=format!("새 수식 시험 · {}",choice.describe());
                self.trace=self.state.frontier.solve(&q,true).trace;
            }
        }
        if self.exiting {self.in_exam=false;return;}
        self.exam_progress=n;self.in_exam=false;
        record.unique=exact.len();record.structural_unique=structures.len();
        record.passed=n==EXAM_SIZE && record.accuracy()>=0.90 &&
            record.step_rate()>=0.90 && record.completion()>=0.90 &&
            record.unique>=n*7/10 && record.structural_unique>=n/100;
        self.state.exams+=1;
        self.state.history.push(record.clone());
        if self.state.history.len()>60 {self.state.history.remove(0);}
        self.state.trained_this_stage=0;
        self.notice(format!("개방형 시험 · 정답 {:.1}% · 고유문항 {} · AST 구조 {} · {}",
            record.accuracy()*100.0,record.unique,record.structural_unique,
            if record.passed{"다음 연구 주기"}else{"약점 재학습"}));
        if record.passed{
            self.state.mastered=true;
            self.state.frontier_passes+=1;
            self.state.frontier.epoch+=1;
            self.notice("개방형 출제 연구 주기 통과 · 문법 탐색 깊이 확장");
        }
        if let Err(e)=self.state.save(){self.notice(format!("자동 저장 오류: {e}"));}
        self.send_snapshot(true);
    }
    pub fn run(mut self) {
        self.send_snapshot(true);
        while !self.exiting {
            match self.rx.try_recv() {
                Ok(cmd)=>self.command(cmd),
                Err(TryRecvError::Empty)=>(),
                Err(TryRecvError::Disconnected)=>break,
            }
            if self.exiting {break;}
            if self.exam_requested {self.examination(EXAM_SIZE);}
            else if self.running {self.train_one();}
            else {
                match self.rx.recv_timeout(Duration::from_millis(100)) {
                    Ok(cmd)=>self.command(cmd),
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected)=>break,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout)=>self.send_snapshot(false),
                }
            }
        }
        let _=self.state.save();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn permanent_frontier_starts_after_29_stages() {
        assert_eq!(FRONTIER_STAGE,29);
        assert_eq!(TOTAL_STAGES,30);
    }
    #[test]
    fn zero_questions_never_passes_and_step_rate_is_real() {
        let r=ExamRecord {stage:0,ordinal:1,questions:100_000,unique:10,correct:95_000,
            completed:95_000,correct_steps:81_000,total_steps:100_000,passed:false,
            per_op:std::array::from_fn(|_|OpRecord::default()),advanced:false,structural_unique:0};
        assert!(r.accuracy()>=0.90);
        assert!(r.step_rate()<0.90);
    }
}
