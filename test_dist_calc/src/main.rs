use std::f64::MAX;

#[derive(Clone, Copy, Debug)]
struct Point<T> {
    x: T,
    y: T,
}

#[derive(Clone, Copy, Debug)]
struct Coord {
    x: f64,
    y: f64,
    z: f64,
}

struct Input {
    camera: [Point<f64>; 3],
    target: [Point<f64>; 3],
}

const SOLUTION_TOL: f64 = 0.0001;
const MAX_ITERATIONS: i32 = 100;

fn divided_differences<F: Fn(f64) -> f64>(f: &F, l: &[f64]) -> f64 {
    match l.len() {
        0 => panic!("invalid divided differences range!"),
        1 => f(l[0]),
        _ => (divided_differences(f, &l[1..l.len()]) - divided_differences(f, &l[0..l.len()-1]))/(l[l.len()-1] - l[0])
    }
}

fn mullers_method<F: Fn(f64) -> f64>(f: F, x0: f64, x1: f64, x2: f64) -> Option<f64> {
    let mut xk3 = x0;
    let mut xk2 = x1;
    let mut xk1 = x2;

    let mut yk1 = f(xk1);
    let mut iteration_count = 0;
    //println!("xk1: {}, yk1: {}", xk1, yk1);

    while yk1.abs() > SOLUTION_TOL && iteration_count < MAX_ITERATIONS {
        let tmp = divided_differences(&f, &[xk1, xk2, xk3]);
        let w = divided_differences(&f, &[xk1, xk2])
            + divided_differences(&f, &[xk1, xk3])
            - divided_differences(&f, &[xk2, xk3]);
        let sgn = if w > 0.0 { 1.0 } else { -1.0 };
        let xk = xk1 - 2.0*yk1/(w + sgn*(w*w - 4.0*yk1*tmp).sqrt());

        xk3 = xk2;
        xk2 = xk1;
        xk1 = xk;
        yk1 = f(xk1);
        //println!("xk1: {}, yk1: {}", xk1, yk1);


        iteration_count += 1;
    }
    if yk1.abs() <= SOLUTION_TOL {
        Some(xk1)
    } else {
        None
    }
}

fn secant_method<F: Fn(f64) -> f64>(f: F, x0: f64, x1: f64) -> Option<f64> {
    let mut x0 = x0;
    let mut x1 = x1;

    let mut y_0 = f(x0);
    let mut y_1 = f(x1);
    //println!("x0: {}, x1: {}, y0: {}, y1: {}", x0, x1, y_0, y_1);

    let mut iteration_count = 0;

    while y_1.abs() > SOLUTION_TOL && iteration_count < MAX_ITERATIONS {
        let x_next = x1 - y_1 * (x1 - x0)/(y_1 - y_0);

        x0 = x1;
        x1 = x_next;

        y_0 = y_1;
        y_1 = f(x1);
        //println!("x0: {}, x1: {}, y0: {}, y1: {}", x0, x1, y_0, y_1);
        
        iteration_count += 1;
    }

    if y_1.abs() <= SOLUTION_TOL {
        Some(x1)
    } else {
        None
    }
}

fn quad_secant_method<F: Fn(&[f64; 4]) -> [f64; 4]>(f: &F, xs0: &[f64; 4], xs1: &[f64; 4]) -> Option<(f64, usize)> {
    let mut xs0: [f64; 4] = *xs0;
    let mut xs1: [f64; 4] = *xs1;
    let mut y_0: [f64; 4] = f(&xs0);
    let mut y_1: [f64; 4] = f(&xs1);

    //println!("x0: {:?}, y0: {:?}, x1: {:?}, y1: {:?}", xs0, y_0, xs1, y_1);

    let mut iteration_count = 0;

    while y_1.iter().all(|y| y.is_nan() || y.abs() > SOLUTION_TOL) && iteration_count < MAX_ITERATIONS {
        let mut x_next: [f64; 4] = [0.0; 4];
        for i in 0..xs0.len() {
            x_next[i] = xs1[i] - 0.5 * y_1[i] * (xs1[i] - xs0[i])/(y_1[i] - y_0[i]);
        }

        xs0 = xs1;
        xs1 = x_next;

        y_0 = y_1;
        y_1 = f(&xs1);
        //println!("x0: {:?}, y0: {:?}, x1: {:?}, y1: {:?}", xs0, y_0, xs1, y_1);
        
        iteration_count += 1;
    }

    y_1.iter()
        .enumerate()
        .filter(|&(idx, y)| y.abs() <= SOLUTION_TOL)
        .next()
        .map(|(idx, y)| (xs1[idx], idx))
}

fn linear_then_secant_method<F: Fn(&[f64; 4]) -> [f64; 4]>(f: &F, lower: &[f64; 4], upper: &[f64; 4]) -> Option<(f64, usize)> {
    const NUM_SEARCH: usize = 16;

    let mut closest = [(0.0, std::f64::MAX); 4];

    for i in 0usize..NUM_SEARCH {
        let mut pos: [f64; 4] = [0.0; 4];
        for (idx, p) in pos.iter_mut().enumerate() {
            *p = lower[idx] + (i as f64)*(upper[idx] - lower[idx])/(NUM_SEARCH as f64);
        }
        let result = f(&pos);
        for j in 0usize..4 {
            if result[j].abs() < closest[j].1 {
                closest[j] = (pos[j], result[j].abs());
            }
        }
        //println!("lin search iteration {}: {:?}", i, pos);
        //println!("{:?} {:?}", result, closest);
    }

    println!("lin search results: {:?}", closest);
    let mut closest1: [f64; 4] = [0.0; 4];
    let mut closest2: [f64; 4] = [0.0; 4];
    for i in 0usize..4 {
        closest1[i] = closest[i].0;
        closest2[i] = closest[i].0 + (upper[i] - lower[i])/(2.0 * (NUM_SEARCH as f64));
    }
    quad_secant_method(&f, &closest1, &closest2)
}

fn test_input(inp: Input) -> Option<[Coord; 3]> {
    let camera_depth = 10.0;
    let camera_scale = 0.1;

    let mut alphas = [0.0; 3];
    let mut betas = [0.0; 3];
    for (idx, ref pt) in inp.camera.iter().enumerate() {
        alphas[idx] = camera_scale*(pt.x as f64)/camera_depth;
        betas[idx] = camera_scale*(pt.y as f64)/camera_depth;
    }

    // make immutable for the rest of the function
    let alphas = alphas;
    let betas = betas;

    let sq = |x| x*x;
    let dist = |a: &Point<f64>, b: &Point<f64>| (sq(a.x - b.x) + sq(a.y - b.y)).sqrt();
    let u = dist(&inp.target[0], &inp.target[1]);
    let v = dist(&inp.target[1], &inp.target[2]);
    let w = dist(&inp.target[0], &inp.target[2]);

    let a1 = sq(alphas[1]) + sq(betas[1]) + 1.0;
    let b1 = alphas[0]*alphas[1] + betas[0]*betas[1] + 1.0;
    let c1 = sq(alphas[0]) + sq(betas[0]) + 1.0;

    let z2discr = |z1| ((sq(b1) - a1*c1)*sq(z1) + a1*sq(u)).sqrt();
    let z2plus = |z1: f64| (z1*b1 + z2discr(z1))/a1;
    let z2minu = |z1: f64| (z1*b1 - z2discr(z1))/a1;

    let a2 = sq(alphas[2]) + sq(betas[2]) + 1.0;
    let b2 = alphas[0]*alphas[2] + betas[0]*betas[2] + 1.0;

    let z3discr = |z1| ((sq(b2) - a2*c1)*sq(z1) + a2*sq(w)).sqrt();
    let z3plus = |z1: f64| (z1*b2 + z3discr(z1))/a2;
    let z3minu = |z1: f64| (z1*b2 - z3discr(z1))/a2;

    let expr = |z2: f64, z3: f64| (sq(alphas[2]) + sq(betas[2]) + 1.0)*sq(z3) 
        - 2.0*(alphas[1]*alphas[2] + betas[1]*betas[2] + 1.0)*z2*z3
        + (sq(alphas[1]) + sq(betas[1]) + 1.0)*sq(z2) - sq(v);
    // solve for this last equation

    /*
    let all_expr = |z1| expr(z2plus(z1), z3plus(z1)) * expr(z2minu(z1), z3plus(z1))
        * expr(z2plus(z1), z3minu(z1)) * expr(z2minu(z1), z3minu(z1));
        */
    let all_exprs = |xs: &[f64; 4]| -> [f64; 4] {
        [
            expr(z2plus(xs[0]), z3plus(xs[0])),
            expr(z2plus(xs[1]), z3minu(xs[1])),
            expr(z2minu(xs[2]), z3plus(xs[2])),
            expr(z2minu(xs[3]), z3minu(xs[3])), 
        ]
    };

    // let maybe_z1 = secant_method(all_expr, 10.0, 20.0);
    // let maybe_z1 = quad_secant_method(&all_exprs, &[10.0; 4], &[12.0; 4]);
    let max_z = ((-a1*sq(u)/(sq(b1)-a1*c1)).sqrt()).
        min((-a2*sq(w)/(sq(b2)-a2*c1)).sqrt());
    let maybe_z1 = linear_then_secant_method(&all_exprs, &[camera_depth; 4], &[max_z; 4]);
    println!("{:?}", maybe_z1);
    match maybe_z1 {
        None => {
            None
        },
        Some((z1, idx)) => {
            let z1 = z1;
            let z2 = if idx == 0 || idx == 1 {
                z2plus(z1)
            } else {
                z2minu(z1)
            };
            let z3 = if idx == 0 || idx == 2 {
                z3plus(z1)
            } else {
                z3minu(z1)
            };

            Some([
                Coord{
                    x: alphas[0]*z1,
                    y: betas[0]*z1,
                    z: z1 - camera_depth,
                },
                Coord{
                    x: alphas[1]*z2,
                    y: betas[1]*z2,
                    z: z2 - camera_depth,
                },
                Coord{
                    x: alphas[2]*z3,
                    y: betas[2]*z3,
                    z: z3 - camera_depth,
                },
            ])
        }
    }
}

fn main() {
    println!("Hello, world!");
    let ret1 = test_input(Input{
        camera: [
            Point{x: 19.0, y: 50.0},
            Point{x: 83.0, y: -19.0},
            Point{x: 19.0, y: -14.0},
        ],
        target: [
            Point{x: 100.0, y: -100.0},
            Point{x: -100.0, y: 100.0},
            Point{x: 100.0, y: 100.0},
        ],
    });
    println!("{:?}", ret1);
    let ret1 = ret1.unwrap();
    let cret1: [Point<f64>; 3] = [
        Point{
            x: ret1[0].x/ret1[0].z,
            y: ret1[0].y/ret1[0].z,
        },
        Point{
            x: ret1[1].x/ret1[1].z,
            y: ret1[1].y/ret1[1].z,
        },
        Point{
            x: ret1[2].x/ret1[2].z,
            y: ret1[2].y/ret1[2].z,
        },
    ];
    println!("{:?}", cret1);
    let ret2 = test_input(Input{
        camera: [
            Point{x: 132.0, y: -22.0},
            Point{x: 65.0, y: 47.0},
            Point{x: 131.0, y: 42.0},
        ],
        target: [
            Point{x: 100.0, y: -100.0},
            Point{x: -100.0, y: 100.0},
            Point{x: 100.0, y: 100.0},
        ],
    });
    println!("{:?}", ret2);
}
