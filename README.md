# REnforce
[![Build Status](https://travis-ci.org/NivenT/REnforce.svg?branch=master)](https://travis-ci.org/NivenT/REnforce)

Reinforcement library written in Rust

This library is still in early stages, and the API has not yet been finalized. The documentation can be found [here](https://nivent.github.io/REnforce/renforce/). Contributions and comments are welcomed.

As things are right now, the main focus has been on getting some working examples to see how the library can be used, to get a feel for how reasonable the API is, and to get more comfortable with RL. Going forward, the API still needs to be improved (made more intuitive and customizable), the code needs to be safer, more RL algorithms need to be incorporated, and documentation needs a lot of work. 

## Adding to project
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
In particular, once an environment has been set up, an agent can be trained and testing using code similar to this

```Rust
fn main() {
	// What actions are available to the agent?
	let action_space = ...
	// Here, the agent will use a table as its Q-function
	let q_func = QTable::new();
	// Creates an epsilon greedy Q-agent
	// Agent will use softmax to act randomly 5% of the time
	let mut agent = EGreedyQAgent::new(Box::new(q_func), action_space, 0.05, Softmax::new(1.0));
    // Create the environment
	let mut env = ...
	// Here, we use Q-learning to train the agent with
	// discount factor and learning rate both 0.9 and
	// 10000 training iterations
	let trainer = QLearner::new(action_space, 0.9, 0.9, 10000);

	// Magic happens
	trainer.train(&mut agent, &mut env);

	// Simulate one episode of the environment to see what the agent learned
	let mut obs = env.reset();
	while !obs.done {
		env.render();

		let action = agent.get_action(obs.state);
		obs = env.step(action);

        // wait for user to press enter
		let _ = stdin().read_line(&mut String::new());
	}
	env.render();
}
```

## Progress
A lot remains to be done, but the following reinforcement learning algorithms have been implemented thus far...

* [Q-learning](https://www.wikiwand.com/en/Q-learning)
* [SARSA](https://www.wikiwand.com/en/State-Action-Reward-State-Action)
