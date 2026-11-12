//! MathNNUE student. At inference time all arithmetic digit transitions are
//! neural predictions, not lookups into the trusted math oracle.
use crate::math::{Expr, Op};
use crate::neural::{softmax, SparseNet};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

const MICRO_FEATURES: usize = 35; // 5 operations + 10 slots each for a,b,carry
const MAX_INPUT: u64 = 1_000_000_000_000_000;
#[derive(Clone, Copy)]
enum MicroOp { Add, Sub, Mul, Div, Cmp }
impl MicroOp {
    fn idx(self) -> usize { match self { Self::Add => 0, Self::Sub => 1, Self::Mul => 2, Self::Div => 3, Self::Cmp => 4 } }
    fn label(self) -> &'static str { match self { Self::Add=>"ADD", Self::Sub=>"SUB", Self::Mul=>"MUL", Self::Div=>"DIV", Self::Cmp=>"CMP" } }
}
fn features(op: MicroOp, a: u8, b: u8, carry: u8) -> [usize; 4] {
    [op.idx(), 5 + a as usize, 15 + b as usize, 25 + carry as usize]
}
fn digits(mut v: u64) -> Vec<u8> {
    let mut result = Vec::new();
    loop {
        result.push((v % 10) as u8);
        v /= 10;
        if v == 0 { break; }
    }
    result
}
fn strip(v: &mut Vec<u8>) {
    while v.len() > 1 && v.last() == Some(&0) { v.pop(); }
}
fn decimal(v: &[u8]) -> Result<u64, String> {
    let mut result = 0_u64;
    for &digit in v.iter().rev() {
        result = result.checked_mul(10).and_then(|n| n.checked_add(digit as u64))
            .ok_or("NNUE 출력 범위를 초과했습니다")?;
    }
    Ok(result)
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Student {
    pub brain: SparseNet,
    pub steps_trained: u64,
}
#[derive(Clone, Debug)]
pub struct Solution {
    pub value: Option<i64>,
    pub error: Option<String>,
    pub nodes: Vec<Option<i64>>, // one result per expression-tree operation (postorder)
    pub trace: Vec<String>,
    pub confidence: f32,
}
impl Solution {
    pub fn complete(&self) -> bool { self.value.is_some() && self.error.is_none() }
}
struct Scratch {
    nodes: Vec<Option<i64>>,
    trace: Vec<String>,
    confidence_sum: f32,
    micro_count: usize,
}

impl Student {
    pub fn new() -> Self { Self { brain: SparseNet::new(MICRO_FEATURES, 96, 20, 20261112), steps_trained: 0 } }

    fn infer_step(&self, op: MicroOp, a: u8, b: u8, c: u8, s: &mut Scratch, verbose: bool) -> (u8,u8) {
        let pass = self.brain.forward(&features(op,a,b,c));
        let probabilities = softmax(&pass.scores[..10]);
        let d = probabilities.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).map(|p| p.0).unwrap_or(0);
        let (carry, certainty) = if matches!(op, MicroOp::Cmp) {
            (0, probabilities[d])
        } else {
            let cp = softmax(&pass.scores[10..20]);
            let next = cp.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1)).map(|p| p.0).unwrap_or(0);
            (next, probabilities[d].min(cp[next]))
        };
        s.micro_count += 1;
        s.confidence_sum += certainty;
        if verbose && s.trace.len() < 40 {
            s.trace.push(format!("{}  a={} b={} state={}  → digit={} next={}  confidence={:.1}%",
                op.label(), a, b, c, d, carry, 100.0 * certainty));
        }
        (d as u8, carry as u8)
    }

    fn learned_compare(&self, a: u64, b: u64, s: &mut Scratch, verbose: bool) -> Result<Ordering,String> {
        let ad = digits(a);
        let bd = digits(b);
        if ad.len() != bd.len() { return Ok(ad.len().cmp(&bd.len())); }
        for i in (0..ad.len()).rev() {
            let (prediction, _) = self.infer_step(MicroOp::Cmp, ad[i], bd[i], 0, s, verbose);
            match prediction {
                0 => return Ok(Ordering::Less),
                1 => (),
                2 => return Ok(Ordering::Greater),
                _ => return Err("비교 NNUE가 유효하지 않은 출력을 냈습니다".into()),
            }
        }
        Ok(Ordering::Equal)
    }

    fn add_digits(&self, a: &[u8], b: &[u8], s: &mut Scratch, verbose: bool) -> Result<Vec<u8>,String> {
        let mut answer = Vec::with_capacity(a.len().max(b.len())+1);
        let mut carry = 0_u8;
        for i in 0..a.len().max(b.len()) {
            let da = *a.get(i).unwrap_or(&0);
            let db = *b.get(i).unwrap_or(&0);
            let (digit, next) = self.infer_step(MicroOp::Add,da,db,carry,s,verbose);
            if next > 1 { return Err("자리올림 NNUE 출력이 0 또는 1이 아닙니다".into()); }
            answer.push(digit);
            carry = next;
        }
        if carry != 0 { answer.push(carry); }
        strip(&mut answer);
        Ok(answer)
    }
    fn subtract_digits(&self, a: &[u8], b: &[u8], s: &mut Scratch, verbose: bool) -> Result<Vec<u8>,String> {
        let mut answer = Vec::with_capacity(a.len());
        let mut borrow = 0_u8;
        for i in 0..a.len().max(b.len()) {
            let da = *a.get(i).unwrap_or(&0);
            let db = *b.get(i).unwrap_or(&0);
            let (digit, next) = self.infer_step(MicroOp::Sub,da,db,borrow,s,verbose);
            if next > 1 { return Err("받아내림 NNUE 출력이 0 또는 1이 아닙니다".into()); }
            answer.push(digit);
            borrow = next;
        }
        if borrow != 0 { return Err("마지막 받아내림이 남았습니다".into()); }
        strip(&mut answer);
        Ok(answer)
    }
    fn multiply_digits(&self, a: &[u8], b: &[u8], s: &mut Scratch, verbose: bool) -> Result<Vec<u8>,String> {
        let mut result = vec![0];
        for (position, &db) in b.iter().enumerate() {
            let mut row = vec![0; position];
            let mut carry = 0;
            for &da in a {
                let (digit, next) = self.infer_step(MicroOp::Mul,da,db,carry,s,verbose);
                row.push(digit);
                carry = next;
            }
            if carry != 0 { row.push(carry); }
            result = self.add_digits(&result,&row,s,verbose)?;
        }
        strip(&mut result);
        Ok(result)
    }
    fn divide_digits(&self, a: &[u8], divisor: u8, s: &mut Scratch, verbose: bool) -> Result<Vec<u8>,String> {
        let mut quotient = Vec::with_capacity(a.len());
        let mut remainder = 0;
        for &digit in a.iter().rev() {
            let (q, r) = self.infer_step(MicroOp::Div,digit,divisor,remainder,s,verbose);
            if r >= divisor { return Err("나눗셈의 나머지 상태가 잘못되었습니다".into()); }
            quotient.push(q);
            remainder = r;
        }
        if remainder != 0 { return Err("정확히 나누어지지 않습니다 (현재 정수 나눗셈만 지원)".into()); }
        quotient.reverse();
        strip(&mut quotient);
        Ok(quotient)
    }

    fn binary(&self, op: Op, a: i64, b: i64, s: &mut Scratch, verbose: bool) -> Result<i64,String> {
        if a.unsigned_abs() > MAX_INPUT || b.unsigned_abs() > MAX_INPUT {
            return Err("현재 NNUE 학습 범위(절댓값 10^15)를 벗어났습니다".into());
        }
        let (magnitude, negative) = match op {
            Op::Add | Op::Sub => {
                let b_negative = if op == Op::Sub { b >= 0 } else { b < 0 };
                let a_negative = a < 0;
                if a_negative == b_negative {
                    (self.add_digits(&digits(a.unsigned_abs()),&digits(b.unsigned_abs()),s,verbose)?, a_negative)
                } else {
                    match self.learned_compare(a.unsigned_abs(), b.unsigned_abs(),s,verbose)? {
                        Ordering::Greater | Ordering::Equal =>
                            (self.subtract_digits(&digits(a.unsigned_abs()),&digits(b.unsigned_abs()),s,verbose)?, a_negative),
                        Ordering::Less =>
                            (self.subtract_digits(&digits(b.unsigned_abs()),&digits(a.unsigned_abs()),s,verbose)?, b_negative),
                    }
                }
            }
            Op::Mul => (self.multiply_digits(&digits(a.unsigned_abs()),&digits(b.unsigned_abs()),s,verbose)?, (a<0) ^ (b<0)),
            Op::Div => {
                if b == 0 || b.unsigned_abs() > 9 {
                    return Err("현재 나눗셈은 한 자리의 0이 아닌 정수로만 가능합니다".into());
                }
                (self.divide_digits(&digits(a.unsigned_abs()),b.unsigned_abs() as u8,s,verbose)?, (a<0) ^ (b<0))
            }
        };
        let abs = decimal(&magnitude)?;
        if abs > MAX_INPUT { return Err("예측값이 지원 범위를 초과했습니다".into()); }
        let signed = i64::try_from(abs).map_err(|_|"정수 출력 범위를 초과했습니다")?;
        Ok(if negative { -signed } else { signed })
    }

    fn rec(&self, expr:&Expr, s:&mut Scratch, verbose:bool) -> Result<i64,String> {
        match expr {
            Expr::Num(n) => Ok(*n),
            Expr::Calc(op,left,right) => {
                let a = self.rec(left,s,verbose);
                let b = self.rec(right,s,verbose);
                let result = match (a,b) {
                    (Ok(a),Ok(b)) => self.binary(*op,a,b,s,verbose),
                    (Err(e),_) | (_,Err(e)) => Err(e),
                };
                s.nodes.push(result.as_ref().ok().copied());
                if verbose && s.trace.len() < 40 {
                    s.trace.push(format!("{} = {}",expr,
                        result.as_ref().map(|v|v.to_string()).unwrap_or_else(|_| "풀이 실패".into())));
                }
                result
            }
        }
    }
    pub fn solve(&self, expr:&Expr, verbose:bool) -> Solution {
        let mut s = Scratch { nodes:Vec::new(),trace:Vec::new(),confidence_sum:0.0,micro_count:0 };
        let output = self.rec(expr,&mut s,verbose);
        Solution {
            value: output.as_ref().ok().copied(),
            error: output.err(),
            nodes: s.nodes,
            trace: s.trace,
            confidence: if s.micro_count==0 {1.0} else {s.confidence_sum / s.micro_count as f32},
        }
    }

    fn teach_step(&mut self, op:MicroOp, a:u8,b:u8,c:u8, digit:u8, next:u8) {
        let active = features(op,a,b,c);
        let pass = self.brain.forward(&active);
        let mut gradient = vec![0.0_f32;20];
        let digit_probs = softmax(&pass.scores[..10]);
        for i in 0..10 { gradient[i] = digit_probs[i]; }
        gradient[digit as usize] -= 1.0;
        if !matches!(op,MicroOp::Cmp) {
            let next_probs = softmax(&pass.scores[10..20]);
            for i in 0..10 { gradient[10+i] = next_probs[i]; }
            gradient[10+next as usize] -= 1.0;
        }
        self.brain.train(&active,&pass,&gradient,0.0025);
        self.steps_trained += 1;
    }
    fn teach_add(&mut self, a:u64,b:u64) {
        let ad=digits(a);let bd=digits(b);
        let mut carry=0;
        for i in 0..ad.len().max(bd.len()) {
            let da=*ad.get(i).unwrap_or(&0);
            let db=*bd.get(i).unwrap_or(&0);
            let total=da as u16 + db as u16 + carry as u16;
            self.teach_step(MicroOp::Add,da,db,carry,(total%10) as u8,(total/10) as u8);
            carry=(total/10) as u8;
        }
    }
    fn teach_sub(&mut self,a:u64,b:u64) {
        debug_assert!(a>=b);
        let ad=digits(a);let bd=digits(b);
        let mut borrow=0;
        for i in 0..ad.len().max(bd.len()) {
            let da=*ad.get(i).unwrap_or(&0);
            let db=*bd.get(i).unwrap_or(&0);
            let difference=da as i16-db as i16-borrow as i16;
            let (result,next) = if difference<0 { ((difference+10) as u8,1) } else { (difference as u8,0) };
            self.teach_step(MicroOp::Sub,da,db,borrow,result,next);
            borrow=next;
        }
    }
    fn teach_mul(&mut self,a:u64,b:u64) {
        let ad=digits(a);let bd=digits(b);
        let mut acc=0_u64;
        for (position,&db) in bd.iter().enumerate() {
            let mut carry=0_u8;
            for &da in &ad {
                let sum=da as u16*db as u16+carry as u16;
                self.teach_step(MicroOp::Mul,da,db,carry,(sum%10) as u8,(sum/10) as u8);
                carry=(sum/10) as u8;
            }
            let row = a * db as u64 * 10_u64.pow(position as u32);
            self.teach_add(acc,row);
            acc += row;
        }
    }
    fn teach_div(&mut self,a:u64,b:u8) {
        let mut remainder=0_u8;
        for &da in digits(a).iter().rev() {
            let n=remainder as u16*10+da as u16;
            let q=(n/b as u16) as u8;
            let next=(n%b as u16) as u8;
            self.teach_step(MicroOp::Div,da,b,remainder,q,next);
            remainder=next;
        }
    }
    fn teach_signed_add(&mut self,a:i64,b:i64) {
        let av=a.unsigned_abs();let bv=b.unsigned_abs();
        if (a<0)==(b<0) { self.teach_add(av,bv); }
        else if av>=bv { self.teach_sub(av,bv); }
        else { self.teach_sub(bv,av); }
        // Additional comparisons prevent the network from guessing signs
        // using a hardcoded numeric comparison at inference time.
        let ad=digits(av); let bd=digits(bv);
        for i in 0..ad.len().min(bd.len()) {
            self.teach_cmp(ad[i],bd[i]);
        }
    }
    fn teach_cmp(&mut self,a:u8,b:u8) {
        let label=if a<b {0} else if a==b {1} else {2};
        self.teach_step(MicroOp::Cmp,a,b,0,label,0);
    }
    fn teach_binary(&mut self,op:Op,a:i64,b:i64) {
        match op {
            Op::Add=>self.teach_signed_add(a,b),
            Op::Sub=>self.teach_signed_add(a,-b),
            Op::Mul=>self.teach_mul(a.unsigned_abs(),b.unsigned_abs()),
            Op::Div=>if b != 0 && b.unsigned_abs()<=9 { self.teach_div(a.unsigned_abs(),b.unsigned_abs() as u8); },
        }
    }
    /// Supervision uses the oracle on the separate teaching path only.
    pub fn learn_problem(&mut self, expr:&Expr) {
        if let Expr::Calc(op,left,right) = expr {
            self.learn_problem(left);
            self.learn_problem(right);
            if let (Ok(a),Ok(b))=(crate::math::oracle(left), crate::math::oracle(right)) {
                self.teach_binary(*op,a,b);
            }
        }
    }
    /// Neural pretraining of *digit transitions*, not a table consulted by inference.
    pub fn drill<R:Rng>(&mut self,rng:&mut R) {
        let a=rng.gen_range(0..10) as u8;
        let b=rng.gen_range(0..10) as u8;
        match rng.gen_range(0..5) {
            0=>{let c=rng.gen_range(0..2) as u8;let n=a as u16+b as u16+c as u16;
                self.teach_step(MicroOp::Add,a,b,c,(n%10) as u8,(n/10) as u8);},
            1=>{let c=rng.gen_range(0..2) as u8;let n=a as i16-b as i16-c as i16;
                let (d,next)=if n<0{((n+10) as u8,1)}else{(n as u8,0)};
                self.teach_step(MicroOp::Sub,a,b,c,d,next);},
            2=>{let c=rng.gen_range(0..9) as u8;let n=a as u16*b as u16+c as u16;
                self.teach_step(MicroOp::Mul,a,b,c,(n%10) as u8,(n/10) as u8);},
            3=>{let d=rng.gen_range(1..10) as u8;let rem=rng.gen_range(0..d);
                let n=rem as u16*10+a as u16;
                self.teach_step(MicroOp::Div,a,d,rem,(n/d as u16) as u8,(n%d as u16) as u8);},
            _=>self.teach_cmp(a,b),
        }
    }
}

/// Grading code never enters `Student::solve`. It runs after predictions.
pub fn expected_nodes(expr:&Expr, out:&mut Vec<Option<i64>>) {
    if let Expr::Calc(_,l,r)=expr {
        expected_nodes(l,out);
        expected_nodes(r,out);
        out.push(crate::math::oracle(expr).ok());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn untrained_model_is_not_an_oracle() {
        let student=Student::new();
        assert_eq!(student.brain.outputs,20);
        let expr=crate::math::parse("1+1").unwrap();
        let sol=student.solve(&expr,true);
        assert_eq!(sol.nodes.len(),1);
        assert!(!sol.trace.is_empty());
    }
    #[test]
    fn decimal_roundtrip() {
        for n in [0,1,10,123456789,1000000000000000_u64] {
            assert_eq!(decimal(&digits(n)).unwrap(),n);
        }
    }
}
