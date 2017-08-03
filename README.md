# REnforce
[![Build Status](https://travis-ci.org/NivenT/REnforce.svg?branch=master)](https://travis-ci.org/NivenT/REnforce)

Reinforcement library written in Rust

This library is still in early stages, and the API has not yet been finalized. The documentation can be found [here](https://nivent.github.io/REnforce/renforce/). Contributions and comments are welcomed.

As things are right now, the main focus has been on getting some working examples to see how the library can be used, to get a feel for how reasonable the API is, and to get more comfortable with RL. Going forward, the API still needs to be improved (made more intuitive and customizable), the code needs to be safer (less prone to panic), more RL algorithms need to be incorporated, and the documentation needs a lot of work. 

## Adding library to your project
Use [cargo](http://doc.crates.io/guide.html) to add this library to your project. This library is not in [crates.io](https://crates.io/) yet, so add the following to your `Cargo.toml` in order to include it
```
[dependencies]
renforce = {git = "https://github.com/NivenT/REnforce.git"}
```
and remember to extern it in your project
```Rust
extern crate renforce;
```

## Example Usage
See the [examples](https://github.com/NivenT/REnforce/tree/master/examples) and [tests](https://github.com/NivenT/REnforce/tree/master/tests) folders for example usage.
In particular, once an environment has been set up, an agent can be trained and tested using code similar to this

```Rust
fn main() {
	// Create the environment
	let mut env = ExampleEnvironment::new();
	
	// Here, the agent will use linear value approximators for each action in a given state
	// The agent will select actions based on how high of a value it assigns them
	let q_func = QLinear::default(&env.action_space());
	
	// Creates an epsilon greedy Q-agent
	// Agent will use softmax to act randomly 5% of the time
	let mut agent = EGreedyQAgent::new(q_func, env.action_space(), 0.05, Softmax::default());
	
	// Here, we use Q-learning to train the agent
	// By default the discount factor is 0.95,
	//            the learning rate is 0.1,
	//            the trainer trains for 100 episodes when called train
	// We set the learning rate (alpha) to 0.9
	let trainer = QLearner::default(&env.action_space()).alpha(0.9);

	// Magic happens
	trainer.train(&mut agent, &mut env);

	// Simulate one episode of the environment to see what the agent learned
	let mut obs = env.reset();
	while !obs.done {
		env.render();

		let action = agent.get_action(obs.state);
		obs = env.step(action);
	}
	env.render();
}
```

## Contributing
If you see something that could be done better, or have an idea for a feature that you want to add, then fork and sumbit a PR and/or create an issue for it. If you want to contribute, but you're not sure where to start, feel free to email me with any questions.

## Progress
A lot remains to be done, but the following reinforcement learning algorithms have been implemented<sup>*</sup> thus far...

* [Q-learning](https://www.wikiwand.com/en/Q-learning)
* [SARSA](https://www.wikiwand.com/en/State-Action-Reward-State-Action)
* [Cross Entropy](https://esc.fnwi.uva.nl/thesis/centraal/files/f2110275396.pdf)
* [Vanilla Policy Gradients](https://youtu.be/PtAIh9KSnjo?t=2590)
* [Natural Evolution Strategies](https://arxiv.org/pdf/1703.03864.pdf)


<sup>*</sup> - implementations possibly flawed, but you have to start somewhere
