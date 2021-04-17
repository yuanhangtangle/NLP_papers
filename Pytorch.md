# PyTorch Notes

> @author:  Yuanhang Tang (汤远航)
>
> @e-mail: yuanhangtangle@gmail.com

## [Tutorial/quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

```mermaid
graph LR
	X,Y --> tt(training_data, test_data)
	tt --> dl(dataloader)
	bs(batch_size) --> dl
	
	tnm(nn.Module) --> model(model)
	init(__init__) --> model
	fw(forward) --> model
	
	model --> cmp(computation)
	cmp --> output(output)
	dl --> cmp
	output --> loss(loss)
	tl(true_label) --> loss
	lf(loss_function) --> loss
	loss --> bw(backward)
	bw --> mp(gradient of model parameters)
	optim --> up(update)
	mp --> up
	up --> zero_grad
	zero_grad --> cmp
```

### Commonly used modules

- torch
- torch.nn
  - nn.Parameter
  - nn.Sequential
- torch.nn.functional
- torch.nn.Module
- torch.utils.data.Dataloader
- torch.optim