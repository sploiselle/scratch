use itertools::Itertools;
use ordered_float::OrderedFloat;
use rand::distributions::{Distribution, Uniform};
use timely::dataflow::channels::pact::Pipeline;
use timely::dataflow::operators::{Input, Operator};
use timely::dataflow::InputHandle;

fn main() {
    timely::execute_from_args(std::env::args(), |worker| {
        let index = worker.index();
        let mut input: timely::dataflow::InputHandle<i32, (Vec<OrderedFloat<f64>>, bool)> =
            InputHandle::new();

        let training_out = worker.dataflow(|scope| {
            scope
                .input_from(&mut input)
                .unary(Pipeline, "mul", |capability, _info| {
                    drop(capability);

                    let mut rng = rand::thread_rng();
                    let init = Uniform::from(-0.01..0.01);
                    let v = vec![OrderedFloat(init.sample(&mut rng)); 3];

                    let mut vector = Vec::new();
                    move |input, output| {
                        while let Some((time, data)) = input.next() {
                            data.swap(&mut vector);
                            let mut session = output.session(&time);
                            for (input, b) in vector.drain(..) {
                                for (v, i) in v.iter_mut().zip_eq(input.into_iter()) {
                                    if b {
                                        *v = OrderedFloat(v.0 * i.0);
                                    }
                                }
                            }

                            session.give(v.clone())
                        }
                    }
                })
                .unary(Pipeline, "arrange", |capability, _info| {
                    drop(capability);

                    move |input, output| {
                        let predictor = input.as_collection(scope);
                        let predictor = predictor.arrange_by_self();
                        predictor.trace
                    }
                })
        });

        // introduce data and watch!
        for round in 0..10 {
            if index == 0 {
                input.send((vec![OrderedFloat(round as f64); 2], true));
            }
            input.advance_to(round + 1);
        }
    })
    .unwrap();
}
