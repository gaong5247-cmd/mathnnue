//! Small, explicit parser for user-supplied equations and polynomial functions.
//! It is deliberately not advertised as a universal CAS.
use crate::advanced::{Challenge,Task};

fn normalise(s:&str)->String {
    s.replace('−',"-").replace('²',"^2").replace('³',"^3").replace('⁴',"^4")
        .replace('×',"*").replace(' ',"").replace('*',"")
}
fn polynomial(s:&str)->Result<[i64;5],String>{
    let s=normalise(s);
    if s.is_empty(){return Err("빈 수식입니다".into());}
    let mut coeffs=[0_i64;5];
    let mut expanded=String::new();
    if !s.starts_with('+') && !s.starts_with('-'){expanded.push('+');}
    expanded.push_str(&s);
    let bytes=expanded.as_bytes();
    let mut start=0;
    for i in 1..=expanded.len() {
        if i<expanded.len() && bytes[i]!=b'+' && bytes[i]!=b'-' {continue;}
        if start==i{continue;}
        let term=&expanded[start..i];
        if term.len()<2{return Err(format!("잘못된 항: {term}"));}
        let sign=if term.starts_with('-'){-1_i64}else{1};
        let t=&term[1..];
        let (degree,weight)=if let Some(ix)=t.find('x'){
            let prefix=&t[..ix];
            let a=if prefix.is_empty(){1}else{prefix.parse::<i64>().map_err(|_|"계수가 잘못됨")?};
            let power=&t[ix+1..];
            let d=if power.is_empty(){1}else{power.strip_prefix('^').ok_or("지수 앞에 ^를 입력하세요")?.parse::<usize>().map_err(|_|"지수 오류")?};
            (d,a)
        } else {(0,t.parse::<i64>().map_err(|_|"숫자 항 오류")?)};
        if degree>4 {return Err("현재는 4차 이하 다항식만 지원합니다".into());}
        coeffs[degree]=coeffs[degree].checked_add(sign*weight).ok_or("계수 초과")?;
        start=i;
    }
    Ok(coeffs)
}
fn task(kind:usize,p:[i64;8])->Challenge {
    Challenge {tasks:vec![Task{kind,p,style:0}],title:"사용자 지정 수식".into(),
        family:kind,difficulty:0,style:0,mutation:0,pattern:0}
}
pub fn parse_custom(input:&str)->Result<Challenge,String>{
    let trimmed=input.trim();
    let normalized=normalise(trimmed);
    if normalized.starts_with("diff") || normalized.starts_with("derive") || trimmed.starts_with("미분") {
        let (expr,at)=at_clause(trimmed)?;
        let expr=expr.trim_start_matches("diff").trim_start_matches("derive").trim_start_matches("미분").trim();
        let expr=expr.trim_start_matches("f(x)=");
        let c=polynomial(expr)?;
        if c[4]!=0{return Err("현재 미분 입력은 3차 다항식까지 지원합니다".into());}
        let mut p=[0;8];p[0]=c[3];p[1]=c[2];p[2]=c[1];p[3]=c[0];p[4]=at;
        return Ok(task(11,p));
    }
    if normalized.starts_with("integral") || trimmed.starts_with("적분") {
        let tokens:Vec<_>=trimmed.split_whitespace().collect();
        if tokens.len()<4{return Err("적분 입력: integral 0 2 3x^2+4x".into());}
        let from=tokens[1].parse::<i64>().map_err(|_|"하한이 정수가 아닙니다")?;
        if from!=0{return Err("정적분은 현재 0부터 시작하는 구간만 지원합니다".into());}
        let upper=tokens[2].parse::<i64>().map_err(|_|"상한이 정수가 아닙니다")?;
        let c=polynomial(&tokens[3..].join(""))?;
        if c[4]!=0 || c[3]!=0 || c[0]!=0{return Err("정적분은 ax²+bx 꼴까지 지원합니다".into());}
        let mut p=[0;8];p[0]=upper;p[1]=c[2];p[2]=c[1];
        return Ok(task(12,p));
    }
    if let Some((left,right))=normalized.split_once('=') {
        let a=polynomial(left)?;
        let b=polynomial(right)?;
        let c:Vec<i64>=(0..5).map(|i|a[i]-b[i]).collect();
        let degree=(0..5).rev().find(|&i|c[i]!=0).ok_or("항등식입니다")?;
        if degree==1 {
            if c[0]%c[1]!=0{return Err("현재 사용자 지정 일차방정식은 정수해만 지원합니다".into());}
            let mut p=[0;8];p[0]=c[1];p[1]=c[0];p[2]=-c[0]/c[1];
            return Ok(task(1,p));
        }
        if (2..=4).contains(&degree) {
            if c[degree]!=1{return Err("현재 다항방정식은 최고차항 계수가 1인 경우만 지원합니다".into());}
            let mut coeffs=c[..=degree].to_vec();
            coeffs.reverse();
            let mut roots=Vec::new();
            for _ in 0..degree {
                let root=(-35..=35).find(|&x|coeffs.iter().fold(0_i64,|v,&k|v*x+k)==0)
                    .ok_or("현재는 모든 근이 작은 정수인 방정식만 지원합니다")?;
                roots.push(root);
                let mut q=vec![coeffs[0]];
                for &v in coeffs.iter().skip(1).take(coeffs.len()-2){q.push(q.last().copied().unwrap()*root+v);}
                coeffs=q;
            }
            let mut p=[0_i64;8];for(i,r)in roots.into_iter().enumerate(){p[i]=r;}
            return Ok(task(degree+1,p));
        }
        return Err("일차~사차 방정식만 지원합니다".into());
    }
    Err("예: 2x+3=7, x^2-5x+6=0, diff x^3+2x at x=2, integral 0 2 3x^2+4x".into())
}
fn at_clause(input:&str)->Result<(&str,i64),String>{
    let (expr,at)=input.split_once(" at x=").ok_or("예: diff x^3-2x at x=3")?;
    let at=at.trim().parse::<i64>().map_err(|_|"x의 대입값은 정수여야 합니다")?;
    Ok((expr,at))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equations_and_calculus(){
        let q=parse_custom("3x+7=22").unwrap();assert_eq!(q.expected(),"5");
        let q=parse_custom("x^2-5x+6=0").unwrap();assert_eq!(q.expected(),"2, 3");
        let q=parse_custom("x³-6x²+11x-6=0").unwrap();assert_eq!(q.expected(),"1, 2, 3");
        let q=parse_custom("diff 2x^3-3x^2+4x+1 at x=2").unwrap();assert_eq!(q.expected(),"16");
        let q=parse_custom("integral 0 2 3x^2+4x").unwrap();assert_eq!(q.expected(),"16");
    }
}
