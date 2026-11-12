mod advanced;
mod creative;
mod input;
mod examiner;
mod math;
mod neural;
mod solver;
mod trainer;

use eframe::egui;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;
use trainer::{Command, Event, Snapshot, Worker, EXAM_SIZE, TRAIN_PER_EXAM};

#[derive(Default)]
struct ManualAnswer {
    question:String,
    prediction:String,
    expected:String,
    confidence:f32,
    trace:Vec<String>,
}
struct MathApp {
    command:Sender<Command>,
    events:Receiver<Event>,
    worker:Option<thread::JoinHandle<()>>,
    snapshot:Snapshot,
    input:String,
    answer:Option<ManualAnswer>,
    quiz:Vec<(String,String)>,
    quiz_count:usize,
    preview_level:usize,
    show_answers:bool,
    notices:Vec<String>,
}
impl MathApp {
    fn new(context:&eframe::CreationContext<'_>)->Self {
        configure_fonts(&context.egui_ctx);
        let (command, worker_commands)=mpsc::channel();
        let (worker_events, events)=mpsc::channel();
        let worker=thread::spawn(move||Worker::new(worker_events,worker_commands).run());
        Self {command,events,worker:Some(worker),snapshot:Snapshot::default(),input:"1+1".into(),
            answer:None,quiz:Vec::new(),quiz_count:10,preview_level:12,show_answers:false,notices:Vec::new()}
    }
    fn send(&self,command:Command) {let _=self.command.send(command);}
    fn poll(&mut self) {
        while let Ok(event)=self.events.try_recv() {
            match event {
                Event::Snapshot(snapshot)=>self.snapshot=snapshot,
                Event::Notice(message)=>{
                    self.notices.push(message);
                    if self.notices.len()>120 {self.notices.remove(0);}
                }
                Event::Answer{question,prediction,expected,confidence,trace}=>{
                    self.answer=Some(ManualAnswer{question,prediction,expected,confidence,trace});
                }
                Event::Quiz(quiz)=>self.quiz=quiz,
            }
        }
    }
    fn heading_panel(&self,ui:&mut egui::Ui) {
        ui.horizontal_wrapped(|ui|{
            ui.heading(egui::RichText::new("MathNNUE").size(30.0).strong());
            ui.label(egui::RichText::new("STUDENT × EXAMINER").color(egui::Color32::LIGHT_BLUE));
            ui.separator();
            ui.label(if self.snapshot.mastered {"인증형 전 과정 통과 · 지속 학습"} else if self.snapshot.in_exam {"10만 문제 시험 진행"}
                else if self.snapshot.running {"학습 중"} else {"일시정지"});
        });
        ui.label("초등: 순수 숫자 예측 NNUE / 고급: 신경망 풀이 전략 + 검증 가능한 수학 규칙 / 출제: 탐색·변이·신규성 보상");
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui|{
            if ui.button("▶ 학습 / 재개").clicked() {self.send(Command::Start);}
            if ui.button("Ⅱ 일시정지").clicked() {self.send(Command::Pause);}
            if ui.button("✓ 100,000문항 시험").clicked() {self.send(Command::Exam);}
            if ui.button("◈ 체크포인트 저장").clicked() {self.send(Command::Save);}
        });
    }
    fn stage_card(&self,ui:&mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui,|ui|{
            ui.heading(format!("현재 단계 {}/{} · {}", self.snapshot.stage+1,trainer::TOTAL_STAGES,self.snapshot.stage_name));
            ui.horizontal_wrapped(|ui|{
                ui.label(format!("학습 문항: {}",fmt_num(self.snapshot.trained)));
                ui.separator();
                ui.label(format!("미세 NNUE 학습: {}",fmt_num(self.snapshot.micro_steps)));
                ui.separator();
                ui.label(format!("출제 NNUE 업데이트: {}",fmt_num(self.snapshot.generation_trials)));
                ui.separator();
                ui.label(format!("최종 단계 재통과: {}",self.snapshot.frontier_passes));
            });
            let fraction=(self.snapshot.trained_this_stage as f32 / TRAIN_PER_EXAM as f32).clamp(0.0,1.0);
            ui.add(egui::ProgressBar::new(fraction).text(format!("이번 단계 학습: {} / {}",fmt_num(self.snapshot.trained_this_stage as u64),fmt_num(TRAIN_PER_EXAM as u64))));
            if self.snapshot.in_exam {
                let frac=(self.snapshot.exam_progress as f32 / EXAM_SIZE as f32).clamp(0.0,1.0);
                ui.add(egui::ProgressBar::new(frac).text(format!("시험 진행: {} / 100,000",fmt_num(self.snapshot.exam_progress as u64))));
            }
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui|{
                ui.label(format!("최근 정답률 {:.1}%",100.0*self.snapshot.recent_accuracy));
                ui.label(format!("최근 완주율 {:.1}%",100.0*self.snapshot.recent_completion));
            });
            ui.label(egui::RichText::new(&self.snapshot.message).weak());
            if self.snapshot.stage>=math::STAGES.len(){
                ui.label(format!("구조 탐색: 최근 {}개 패턴 / 누적 새 패턴 {} / 출제 {}",
                    fmt_num(self.snapshot.novelty_recent as u64),fmt_num(self.snapshot.novelty_fresh),fmt_num(self.snapshot.novelty_total)));
            }
        });
    }
    fn live_card(&self,ui:&mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui,|ui|{
            ui.heading("지금 NNUE가 풀고 있는 문제");
            ui.label(egui::RichText::new(if self.snapshot.current_question.is_empty(){"학습 시작 후 표시됩니다"}
                else {&self.snapshot.current_question}).monospace().size(20.0));
            ui.label(format!("출제 신경망 선택: {}",self.snapshot.neural_policy));
            ui.horizontal_wrapped(|ui|{
                ui.label(egui::RichText::new(format!("풀이 NNUE: {}",self.snapshot.current_guess)).strong());
                ui.label(format!("검증 정답: {}",self.snapshot.expected));
            });
            ui.separator();
            ui.label("신경망의 실제 숫자별 예측 (교사용 정답과 분리)");
            egui::ScrollArea::vertical().max_height(195.0).show(ui,|ui|{
                for step in &self.snapshot.trace {ui.monospace(step);}
            });
        });
    }
    fn exam_card(&self,ui:&mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui,|ui|{
            ui.heading("검증 시험");
            if let Some(record)=&self.snapshot.last_exam {
                ui.label(format!("제{}회 · 단계 {} · {}문항",record.ordinal,record.stage+1,fmt_num(record.questions as u64)));
                ui.horizontal_wrapped(|ui|{
                    ui.label(egui::RichText::new(format!("정답률 {:.2}%",record.accuracy()*100.0)).strong());
                    ui.label(format!("풀이 단계 정확도 {:.2}%",record.step_rate()*100.0));
                    ui.label(format!("완주율 {:.2}%",record.completion()*100.0));
                });
                ui.label(format!("서로 다른 식: {}개",fmt_num(record.unique as u64)));
                if record.advanced {ui.label(format!("구조적으로 구별한 패턴: {}개",fmt_num(record.structural_unique as u64)));}
                ui.label(if record.passed {"결과: 90/90/90 기준 통과 및 다음 학년 승급"}
                    else {"결과: 기준 미달 — 현 단계 반복 학습"});
                ui.separator();
                for (index,op) in ["+","−","×","÷"].iter().enumerate() {
                    let title=if record.advanced {format!("검증 강도 그룹 {}",index+1)} else {format!("{} 연산",op)};
                    let stat=&record.per_op[index];
                    if stat.total==0 {continue;}
                    ui.label(format!("{}: {}문항 · 정답 {:.1}% · 단계 {:.1}%",title,fmt_num(stat.total as u64),
                        100.0*stat.correct as f32/stat.total as f32,
                        if stat.total_steps==0 {0.0}else{100.0*stat.correct_steps as f32/stat.total_steps as f32}));
                }
            } else {
                ui.label("첫 20,000개 학습 문제를 마치면 출제 NNUE가 10만 문제 시험을 실행합니다.");
            }
            ui.add_space(8.0);
            ui.label("시험 기록 · 가로: 시험 순서 / 세로: 정답률");
            score_chart(ui,&self.snapshot.history);
        });
    }
    fn interactive_card(&mut self,ui:&mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui,|ui|{
            ui.heading("직접 문제 보내서 NNUE 시험하기");
            ui.label("예: 1+1, 3x+7=22, x²-5x+6=0, diff 2x^3+4x at x=2, integral 0 2 3x^2+4x");
            ui.horizontal(|ui|{
                let response=ui.add(egui::TextEdit::singleline(&mut self.input).desired_width(f32::INFINITY));
                let submit=ui.button("풀어봐").clicked() ||
                    (response.lost_focus() && ui.input(|i|i.key_pressed(egui::Key::Enter)));
                if submit && !self.input.trim().is_empty() {self.send(Command::Ask(self.input.clone()));}
            });
            if let Some(answer)=&self.answer {
                ui.separator();
                ui.monospace(&answer.question);
                ui.label(egui::RichText::new(format!("NNUE 예측: {}",answer.prediction)).strong());
                ui.label(format!("독립 검증기: {}",answer.expected));
                ui.label(format!("평균 신뢰도(비보정): {:.1}%",answer.confidence*100.0));
                egui::ScrollArea::vertical().max_height(130.0).show(ui,|ui|{
                    for line in &answer.trace {ui.monospace(line);}
                });
            }
            ui.add_space(7.0);
            ui.label("초등은 숫자별 NNUE 예측, 대수·미적분은 NNUE가 풀이 프로그램을 선택합니다. 미지원 자유형 수식도 있습니다.");
        });
    }
    fn creative_card(&mut self,ui:&mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui,|ui|{
            ui.heading("창의 출제 실험실 · 대학/올림피아드/연구형");
            ui.label("미리보기는 학습 승급을 건너뛰지 않습니다. 문제 문법은 확대되지만 실제 IMO 독창성은 검증되지 않았습니다.");
            ui.add(egui::Slider::new(&mut self.preview_level,0..=advanced::STAGES.len()-1).text("미리보기 단계"));
            ui.strong(advanced::STAGES[self.preview_level].name);
            ui.horizontal_wrapped(|ui|{
                if ui.button("이 단계에서 새 문제 조합").clicked(){self.send(Command::Preview(self.preview_level,self.quiz_count));}
                ui.label(format!("상위 문제 영역 {}개 · 새 구조 {}회 · 반복 구조 {}회",
                    advanced::SKILL_COUNT,fmt_num(self.snapshot.novelty_fresh),
                    fmt_num(self.snapshot.novelty_total.saturating_sub(self.snapshot.novelty_fresh))));
            });
        });
    }
    fn quiz_card(&mut self,ui:&mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui,|ui|{
            ui.heading("문제 내는 NNUE · 나에게 시험지 출제");
            ui.horizontal_wrapped(|ui|{
                ui.add(egui::Slider::new(&mut self.quiz_count,1..=40).text("문제 수"));
                if ui.button("새 시험지 만들기").clicked() {self.send(Command::Generate(self.quiz_count));}
                ui.checkbox(&mut self.show_answers,"정답 공개");
            });
            egui::ScrollArea::vertical().max_height(220.0).show(ui,|ui|{
                for (i,(question,answer)) in self.quiz.iter().enumerate() {
                    ui.horizontal_wrapped(|ui|{
                        ui.monospace(format!("{:02}.  {}",i+1,question));
                        if self.show_answers {ui.label(format!("정답: {answer}"));}
                    });
                }
            });
        });
    }
}
impl eframe::App for MathApp {
    fn update(&mut self,ctx:&egui::Context,_frame:&mut eframe::Frame) {
        self.poll();
        egui::CentralPanel::default().show(ctx,|ui|{
            egui::ScrollArea::vertical().auto_shrink([false,false]).show(ui,|ui|{
                self.heading_panel(ui);
                ui.add_space(10.0);
                self.stage_card(ui);
                ui.add_space(6.0);
                ui.columns(2,|cols|{
                    self.live_card(&mut cols[0]);
                    self.exam_card(&mut cols[1]);
                });
                ui.add_space(6.0);
                ui.columns(2,|cols|{
                    self.interactive_card(&mut cols[0]);
                    self.quiz_card(&mut cols[1]);
                });
                ui.add_space(7.0);
                self.creative_card(ui);
                ui.add_space(7.0);
                ui.collapsing("최근 활동 / 구조 설명",|ui|{
                    ui.label(format!("산술 Student/Examiner {} / {} 파라미터",
                        fmt_num(self.snapshot.student_params as u64),fmt_num(self.snapshot.examiner_params as u64)));
                    ui.label(format!("고급 Student/Creative Examiner {} / {} 파라미터",
                        fmt_num(self.snapshot.advanced_params as u64),fmt_num(self.snapshot.creative_params as u64)));
                    ui.label("산술은 신경망 숫자 예측. 고급은 신경망 정책 선택 뒤 명시적 수학 연산 실행(순수 신경망 증명기가 아님).");
                    ui.label("시험 100,000문항의 정답률·풀이 단계율·완주율을 따로 검증. 고유 문제/구조 수 기록.");
                    ui.label("고급 단계의 '인증'은 등록된 문제 문법 범위에 한정됩니다. 실제 IMO 대회 성적이나 새로운 정리 증명을 의미하지 않습니다.");
                    for notice in self.notices.iter().rev().take(18) {ui.label(notice);}
                });
            });
        });
        ctx.request_repaint_after(Duration::from_millis(100));
    }
}
impl Drop for MathApp {
    fn drop(&mut self) {
        let _=self.command.send(Command::Shutdown);
        if let Some(worker)=self.worker.take() { let _=worker.join(); }
    }
}

fn fmt_num(n:u64)->String {
    let s=n.to_string();
    let mut result=String::new();
    for (i,c) in s.chars().enumerate() {
        if i>0 && (s.len()-i)%3==0 {result.push(',');}
        result.push(c);
    }
    result
}
fn score_chart(ui:&mut egui::Ui,history:&[trainer::ExamRecord]) {
    let (rect,_) = ui.allocate_exact_size(egui::vec2(ui.available_width().max(170.0),115.0),egui::Sense::hover());
    let painter=ui.painter_at(rect);
    painter.rect_filled(rect,4.0,egui::Color32::from_gray(28));
    let goal_y=rect.bottom()-0.9*rect.height();
    painter.line_segment([egui::pos2(rect.left(),goal_y),egui::pos2(rect.right(),goal_y)],
        egui::Stroke::new(1.0_f32,egui::Color32::GRAY));
    if history.is_empty(){return;}
    let point=|i:usize,value:f32|{
        let x=if history.len()==1{0.5}else{i as f32/(history.len()-1) as f32};
        egui::pos2(rect.left()+x*rect.width(),rect.bottom()-value.clamp(0.0,1.0)*rect.height())
    };
    for i in 1..history.len(){
        painter.line_segment([point(i-1,history[i-1].accuracy()),point(i,history[i].accuracy())],
            egui::Stroke::new(2.0_f32,egui::Color32::LIGHT_GREEN));
        painter.line_segment([point(i-1,history[i-1].step_rate()),point(i,history[i].step_rate())],
            egui::Stroke::new(1.5_f32,egui::Color32::LIGHT_BLUE));
    }
    for (i,entry) in history.iter().enumerate(){
        painter.circle_filled(point(i,entry.accuracy()),2.5,egui::Color32::LIGHT_GREEN);
    }
}
fn configure_fonts(ctx:&egui::Context) {
    let paths=[
        "C:\\Windows\\Fonts\\malgun.ttf",
        "C:\\Windows\\Fonts\\malgunsl.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ];
    for path in paths {
        if let Ok(bytes)=std::fs::read(path) {
            let mut fonts=egui::FontDefinitions::default();
            fonts.font_data.insert("korean".into(),egui::FontData::from_owned(bytes).into());
            for family in [egui::FontFamily::Proportional,egui::FontFamily::Monospace] {
                if let Some(names)=fonts.families.get_mut(&family) {names.insert(0,"korean".into());}
            }
            ctx.set_fonts(fonts);
            break;
        }
    }
}
fn run_headless(rounds:u64) {
    let (tx,rx)=mpsc::channel();
    let (events,snapshots)=mpsc::channel();
    let handle=thread::spawn(move||Worker::new(events,rx).run());
    let _=tx.send(Command::Start);
    let mut initial:Option<u64>=None;
    let target=rounds.max(1);
    for event in snapshots {
        match event {
            Event::Notice(s)=>println!("{s}"),
            Event::Snapshot(s)=>{
                if initial.is_none(){initial=Some(s.last_exam.as_ref().map(|r|r.ordinal).unwrap_or(0));}
                if let Some(last)=s.last_exam {
                    if last.ordinal>=initial.unwrap_or(0)+target {
                        println!("[EXAM] {} correct {:.2}% steps {:.2}% completion {:.2}%",
                            last.questions,last.accuracy()*100.0,last.step_rate()*100.0,last.completion()*100.0);
                        break;
                    }
                }
            }
            _=>(),
        }
    }
    let _=tx.send(Command::Shutdown);
    let _=handle.join();
}
fn main()->eframe::Result<()> {
    let args:Vec<String>=std::env::args().collect();
    if args.iter().any(|a|a=="--smoke") {
        use rand::{rngs::SmallRng,SeedableRng};
        let mut rng=SmallRng::seed_from_u64(0x20261112);
        let mut generated=0_usize;
        for (stage,profile) in advanced::STAGES.iter().enumerate() {
            for j in 0..100 {
                let family=profile.skills[j%profile.skills.len()];
                let challenge=advanced::generate(stage,family,j%6,j%4,j%8,&mut rng);
                assert!(challenge.certified(),"{}",challenge.text());
                for t in &challenge.tasks {
                    assert_eq!(advanced::execute_program(t,t.kind),Some(t.expected()));
                }
                generated+=1;
            }
        }
        println!("SMOKE OK: {} verified generated challenges / {} topic families",generated,advanced::SKILL_COUNT);
        return Ok(());
    }
    if args.iter().any(|a|a=="--headless") {
        let rounds=args.windows(2).find(|a|a[0]=="--rounds")
            .and_then(|a|a[1].parse::<u64>().ok()).unwrap_or(1);
        run_headless(rounds);
        return Ok(());
    }
    let options=eframe::NativeOptions {
        viewport:egui::ViewportBuilder::default().with_inner_size([1280.0,840.0]).with_min_inner_size([880.0,600.0]),
        ..Default::default()
    };
    eframe::run_native("MathNNUE · Self-Evolving Math Lab",options,Box::new(|cc|Ok(Box::new(MathApp::new(cc)))))
}
