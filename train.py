import random
import time
import math
import torch
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from datahelper import *
from train import *
from models import *
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 19
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def show_result(score, loss):  
	plt.figure(figsize=(10, 6))
	
	plt.ylabel("Score")
	plt.xlabel("Epochs")
	plt.title("BLEU Score Curve", fontsize=18)
	plt.plot(score)

	plt.show()
	
	plt.figure(figsize=(10, 6))
	
	plt.ylabel("Loss")
	plt.xlabel("Epochs")
	plt.title("Loss Curve", fontsize=18)
	plt.plot(loss)

	plt.show()

#compute BLEU-4 score
def compute_bleu(output, reference):
	cc = SmoothingFunction()
	if len(reference) == 3:
		weights = (0.33,0.33,0.33)
	else:
		weights = (0.25,0.25,0.25,0.25)
	return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, 
          criterion, teacher_forcing_ratio, max_length=MAX_LENGTH):
    batch_size = input_tensor.size(0)
    encoder_hidden = encoder.initHidden(batch_size)

    # transpose tensor from (batch_size, seq_len) to (seq_len, batch_size)
    input_tensor = input_tensor.transpose(0, 1)
    target_tensor = target_tensor.transpose(0, 1)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # calculate number of time step
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    
    loss = 0

    #----------sequence to sequence part for encoder----------#
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

    decoder_input = torch.tensor([[SOS_token] for i in range(batch_size)], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	

    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden) 
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden) 
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, vocab, n_iters, print_every=1000, plot_every=100, 
               batch_size=32, learning_rate=0.01, teacher_forcing_ratio=1.0):
    start = time.time()
    plot_losses = []
    plot_scores = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # create dataloader
    trainset = SpellingLoader('train', vocab)
    trainloader = data.DataLoader(trainset, batch_size, shuffle = True)

    criterion = nn.CrossEntropyLoss()
    
    max_score = 0.8785

    for iter in range(1, n_iters + 1):
        for input_tensor, target_tensor in trainloader:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio)
            print_loss_total += loss
            plot_loss_total += loss
            
        # evaluate and save model
        _, avg_bleu = evaluate(encoder, decoder, vocab)
        plot_scores.append(avg_bleu)
        if avg_bleu > max_score:
            max_score = avg_bleu
            print ("Model save...")
            torch.save(encoder, "./models/encoder_{:.4f}.ckpt".format(avg_bleu))
            torch.save(decoder, "./models/decoder_{:.4f}.ckpt".format(avg_bleu))

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) Loss: %.4f BLEU: %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, avg_bleu))
 
        plot_losses.append(plot_loss_total)
        plot_loss_total = 0
            
    print ("The highest score is %s"%max_score)
            
    return plot_scores, plot_losses


# print the prediction and return the bleu score
def show_prediction(inputs, prediction, targets, plot_pred):

	bleu_total = 0
	for idx in range(len(inputs)):
		bleu_total += compute_bleu(prediction[idx], targets[idx])
		if plot_pred:
			output = "\ninput:  {}\ntarget: {}\npred:   {}".format(inputs[idx], targets[idx], prediction[idx])
			print ("="*30+output)
		
	return bleu_total/len(inputs)

def evaluate(encoder, decoder, vocab, batch_size=64, plot_pred=False):
	# create dataloader
	testset = SpellingLoader('test', vocab)
	testloader = data.DataLoader(testset, batch_size=64)
		
	prediction = []
	targets = []
	inputs = []

	with torch.no_grad():    
		for input_tensor, target_tensor in testloader:
			batch_size = input_tensor.size(0)
			encoder_hidden = encoder.initHidden(batch_size)
			
			# convert input&target into string
			for idx in range(batch_size):
				targets.append(vocab.indices2word(target_tensor[idx].data.numpy()))
				inputs.append(vocab.indices2word(input_tensor[idx].data.numpy()))

			input_tensor = input_tensor.to(device)
			target_tensor = target_tensor.to(device)
			
			# transpose tensor from (batch_size, seq_len) to (seq_len, batch_size)
			input_tensor = input_tensor.transpose(0, 1)
			target_tensor = target_tensor.transpose(0, 1)

			# calculate number of time step
			input_length = input_tensor.size(0)
			target_length = target_tensor.size(0)
			
			#----------sequence to sequence part for encoder----------#
			for ei in range(input_length):
				encoder_output, encoder_hidden = encoder(
					input_tensor[ei], encoder_hidden)
		
			decoder_input = torch.tensor([SOS_token for i in range(batch_size)], device=device)
			output = torch.zeros(target_length, batch_size)
			
			decoder_hidden = encoder_hidden
			
			#----------sequence to sequence part for decoder----------#
			for di in range(target_length):
				decoder_output, decoder_hidden = decoder(
					decoder_input, decoder_hidden)
				topv, topi = decoder_output.topk(1)
				decoder_input = topi.squeeze().detach()  # detach from history as input
				output[di] = decoder_input

			# get predict indices
			output = output.transpose(0, 1)    
				
			# convert indices into string
			for idx in range(batch_size):
				prediction.append(vocab.indices2word(output[idx].data.numpy()))
				
		# calculate average BLEU score
		avg_bleu = show_prediction(inputs, prediction, targets, plot_pred)

				
		return prediction, avg_bleu