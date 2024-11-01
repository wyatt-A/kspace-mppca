use std::collections::btree_map::RangeMut;

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

pub fn extract_data(idx:&[usize],readers:&[CflReader]) -> Array2<Complex32> {
    let mut result = Array2::<Complex32>::zeros((idx.len(),readers.len()).f());
    result.axis_iter_mut(Axis(1)).into_par_iter().zip(readers.par_iter()).for_each(|(mut col,reader)|{
        reader.read_into(&idx, col.as_slice_memory_order_mut().unwrap()).unwrap();
    });
    result
}

pub fn insert_data(idx:&[usize], writers:&mut [CflWriter], matrix:&Array2<Complex32>) {
    writers.par_iter_mut().zip(matrix.axis_iter(Axis(1)).into_par_iter()).for_each(|(w,col)|{
        w.write_from(idx, col.as_slice_memory_order().unwrap()).unwrap();
    });
}

/// find the low-rank approximation of a casorati matrix based on known noise levels
/// singular values are sorted greatest to least
fn mp_find_rank(singular_values:&[f32],sigma_sqr:f32,m:usize,n:usize) -> usize {
    assert!(m >= n,"casorati matrix must have at least as many rows as columns");
    let mut rank:usize = n;
    let mut cumulative_sum = 0.;
    for (i,val) in singular_values.iter().rev().enumerate() {
        let scale = (m - n + i - 1) as f32 / n as f32;
        let lamb = val.powi(2) / n as f32;
        cumulative_sum += lamb;
        let cumulative_mean = cumulative_sum / (i + 1) as f32;
        let var = cumulative_mean / scale;
        if var < sigma_sqr {
            rank -= 1;
        }
    }
    rank
}

#[test]
fn test_find_rank() {
    let singular_values:Vec<f32> = vec![267.922179,77.677573,77.367026,76.901532,76.218652,75.661715,74.796348,74.245850,73.357911,73.255173,72.823380,72.246012,71.902821,71.266427,70.512379,70.128508,69.673704,69.044314,68.675660,68.103519,67.717366,67.562810,66.985499,66.324698,66.120651,65.819462,65.727307,65.209190,64.870015,64.603728,64.274472,63.772005,63.499366,62.813258,62.612546,62.395864,61.785515,61.286978,60.677350,60.473087,60.263632,60.187295,59.388814,59.090708,58.686399,58.106206,57.738611,57.491265,57.080259,56.797701,56.341278,55.811282,55.283017,55.040263,53.932982,53.506302,53.326709,52.902144,52.633008,51.825234,51.082119,50.702511,50.321345,49.923459,49.815425,48.324251,47.976007];
    //let rank = find_rank(&singular_values, 4., 1000, 67);
    let (rank,noise) = mp_estimate_rank(&singular_values, 1000, 67);
    println!("rank = {}",rank);
    println!("noise = {}",noise);
}

// estimates the rank and variance simultaneousely
fn mp_estimate_rank(singular_values:&[f32],m:usize,n:usize) -> (usize,f32) {
    assert!(m >= n,"casorati matrix must have at least as many rows as columns");
    let mut rank:usize = n;
    let mut cumulative_sum = 0.;
    let mut noise_estimate = 0.;
    for (i,val) in singular_values.iter().rev().enumerate() {
        let scale = (m - n + i - 1) as f32 / n as f32;
        let lamb = val.powi(2) / n as f32;
        let lamb_last = singular_values.last().unwrap().powi(2) / n as f32;
        cumulative_sum += lamb;
        let cumulative_mean = cumulative_sum / (i + 1) as f32;
        let var = cumulative_mean / scale;
        let gamma = (m - ((n-1) - i)) as f32 / n as f32;
        let range_mp = 4. * gamma.sqrt();
        let range_data = lamb - lamb_last;
        let sig_estimate = range_data / range_mp;
        if sig_estimate < var {
            rank -= 1;
            noise_estimate = var;
        }
    }
    (rank,noise_estimate)
}



struct PCAInfo {
    init_energy:f32,
    final_energy:f32,
    rank:usize,
    variance:f32
}

fn mp_denoise_matrix(matrix:&mut Array2<Complex32>,variance:Option<f32>) -> PCAInfo {

    // m is the number of features (rows)
    let m = matrix.shape()[0];
    // n is the number observations (columns)
    let n = matrix.shape()[1];

    // get the initial energy of the matrix prior to noise reduction
    let init_energy:f32 = matrix.par_iter().map(|x| x.norm_sqr()).sum();

    // calculate SVD
    let (u,mut s,v) = matrix.svddc(ndarray_linalg::UVTFlag::Some).unwrap();
    let u = u.unwrap();
    let v = v.unwrap();

    // retrieve mutable reference to the singular values
    let singular_values = s.as_slice_mut().unwrap();

    // determine the rank and noise (if applicable) of the matrix
    let (rank,var) = match &variance {
        Some(var) => {
            let rank = mp_find_rank(singular_values, *var, m, n);
            (rank,*var)
        }
        None => {
            let (rank,var_estimate) = mp_estimate_rank(singular_values, m, n);
            (rank,var_estimate)
        }
    };

    // do hard thresholding by nullifying singular values above estimated rank
    singular_values[rank..].fill(0.);

    // reconstruct full diagonal matrix from nullified singular values
    let s = Array2::from_diag(&s.map(|&x|Complex32::new(x, 0.)));

    // reconstruct data matrix and assign values to original
    let recon = u.dot(&s).dot(&v);
    matrix.assign(&recon);

    // find the resulting energy of the denoised matrix
    let final_energy:f32 = matrix.par_iter().map(|x| x.norm_sqr()).sum();

    // record meta data and return
    PCAInfo {
        init_energy,
        final_energy,
        rank,
        variance: var,
    }

}

#[test]
fn test_denoise_matrix() {
    
}