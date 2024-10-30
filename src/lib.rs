use cfl::{ndarray::{parallel::prelude::*, Array2, Axis, ShapeBuilder}, num_complex::Complex32, CflReader, CflWriter};
use ndarray_linalg::SVDDC;

pub struct DenoiseInfo {
    pub singular_values:Vec<Complex32>,
    pub estimated_rank:usize,
    pub estimated_noise:Option<f32>,
    pub init_energy:f32,
    pub final_energy:f32,
}

pub fn singular_value_threshold_mppca(matrix:&mut Array2<Complex32>, rank:Option<usize>) -> DenoiseInfo {

        let m = matrix.shape()[0];
        let n = matrix.shape()[1];

        let init_energy = matrix.iter().map(|x|x.norm_sqr()).sum::<f32>();

        //let mut _s = Array2::from_elem((m,n), Complex32::ZERO);

        let nn = m.min(n);
        let mut _s = Array2::from_elem((nn,nn), Complex32::ZERO);

        //let (_,s,_) = matrix.svd(true, true).unwrap();
        // let (_,sigma_sq) = marchenko_pastur_singular_value(&s.as_slice().unwrap(), m, n);
        // matrix.par_mapv_inplace(|x| x / sigma_sq);
        //let (u,mut s,v) = matrix.svd(true, true).unwrap();

        let (u,mut s,v) = matrix.svddc(ndarray_linalg::UVTFlag::Some).unwrap();

        let u = u.unwrap();
        let v = v.unwrap();

        // println!("u:{:?}",u.shape());
        // println!("s:{:?}",s.shape());
        // println!("v:{:?}",v.shape());

        let mut noise = None;

        let rank = rank.unwrap_or_else(||{
            let (rank,sigma_sq) = marchenko_pastur_singular_value(&s.as_slice().unwrap(), m, n);
            noise = Some(sigma_sq);
            rank
        });
        
        let singular_values:Vec<_> = s.iter().map(|&s|Complex32::new(s, 0.)).collect();

        s.iter_mut().enumerate().for_each(|(i,val)| if i >= rank {*val = 0.});

        //let u = u.unwrap();
        //let v = v.unwrap();
        let mut diag_view = _s.diag_mut();
        diag_view.assign(
            &s.map(|x|Complex32::new(*x,0.))
        );

        // if u in 1x1
        let denoised_matrix = if u.len() == 1 {
            _s.dot(&v)
        }else {
            u.dot(&_s).dot(&v)
        };

        //let denoised_matrix = u.dot(&_s).dot(&v);
        matrix.assign(&denoised_matrix);

        let final_energy = matrix.iter().map(|x|x.norm_sqr()).sum::<f32>();

        //let energy_makeup = (init_energy / final_energy).sqrt();
        //matrix.iter_mut().for_each(|x| *x *= energy_makeup);

        DenoiseInfo {
            singular_values: singular_values,
            estimated_rank: rank,
            estimated_noise: noise,
            init_energy,
            final_energy,
        }
}

/// returns the index of the singular value to threshold, along with the estimated matrix variance
fn marchenko_pastur_singular_value(singular_values:&[f32],m:usize,n:usize) -> (usize,f32) {
    
    let r = m.min(n);
    let mut vals = singular_values.to_owned();

    let scaling:Vec<_> = (0..r).clone().map(|x| (m.max(n) as f32 - x as f32) / n as f32).collect();
    
    vals.iter_mut().for_each(|x| *x = x.powi(2) / n as f32);
    vals.reverse();
    let mut csum = cumsum(&vals);
    csum.reverse();
    vals.reverse();

    let cmean:Vec<_> = csum.iter().zip((1..r+1).rev()).map(|(x,y)| x / y as f32).collect();
    let sigmasq_1:Vec<_> = cmean.iter().zip(scaling.iter()).map(|(x,y)| *x / *y).collect();
    let range_mp:Vec<_> = (0..r).map(|x| x as f32).map(|x| (m as f32 - x) /  n as f32).map(|x| x.sqrt() * 4.).collect();
    let range_data:Vec<_> = vals[0..r].iter().map(|x| x - vals[r-1]).collect();
    let sigmasq_2:Vec<_> = range_data.into_iter().zip(range_mp).map(|(x,y)|x / y).collect();

    //println!("{:?}",sigmasq_1);
    //println!("{:?}",sigmasq_2);

    let idx = sigmasq_1.iter().zip(sigmasq_2).enumerate().find_map(|(i,(s1,s2))|{
        if s2 < *s1 {
            Some(i)
        }else {
            None
        }
    }).unwrap();

    let variance_estimate = sigmasq_1[idx];

    (idx,variance_estimate)
}

fn cumsum(x:&[f32]) -> Vec<f32> {
    let mut x = x.to_owned();
    let mut s = 0.;
    x.iter_mut().for_each(|val|{
        *val += s;
        s = *val;
    });
    x
}

fn extract_data(idx:&[usize],readers:&[CflReader]) -> Array2<Complex32> {
    let mut result = Array2::<Complex32>::zeros((idx.len(),readers.len()).f());
    result.axis_iter_mut(Axis(1)).into_par_iter().zip(readers.par_iter()).for_each(|(mut col,reader)|{
        reader.read_into(&idx, col.as_slice_memory_order_mut().unwrap()).unwrap();
    });
    result
}

fn insert_data(idx:&[usize], writers:&mut [CflWriter], matrix:&Array2<Complex32>) {
    writers.par_iter_mut().zip(matrix.axis_iter(Axis(1)).into_par_iter()).for_each(|(w,col)|{
        w.write_from(idx, col.as_slice_memory_order().unwrap()).unwrap();
    });
}