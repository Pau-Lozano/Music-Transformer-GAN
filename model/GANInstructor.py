import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from dataset.e_piano import create_epiano_datasets, compute_epiano_accuracy
from utilities.argument_funcs import parse_train_args, print_train_args, write_model_params, parse_eval_args, parse_generate_args
import torch.nn as nn
from model.Generator import Generator
from model.Discriminator import Discriminator
from utilities.constants import *
from utilities.device import get_device
#from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

import torch.nn.functional as F
import random


class GANInstructor:
	def __init__(self, dis_path = None):
		self.train_args = parse_train_args()
		self.eval_args = parse_eval_args()
		self.parse_args = parse_generate_args()

		self.dis_path = dis_path
		self.gen = Generator(n_layers=self.train_args.n_layers, num_heads=self.train_args.num_heads,
							 d_model=self.train_args.d_model, dim_feedforward=self.train_args.dim_feedforward,
							 max_sequence=self.train_args.max_sequence, rpr=self.eval_args.rpr).to(get_device())
		self.dis = Discriminator().to(get_device())
		self.dis_pretrain_criterion = nn.BCELoss().to(get_device())

		self.loadGenerator()
		self.loadDiscriminator()
		self.gen_opt = Adam(self.gen.parameters(), lr=0.00001, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
		self.dis_opt = Adam(self.dis.parameters(), lr=0.00001, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)

		self.train_dataset, self.val_dataset, self.test_dataset = create_epiano_datasets(INPUT_DIR, max_seq=64)
		self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_args.batch_size, num_workers=self.train_args.n_workers, shuffle=True, drop_last=True)
		self.val_loader = DataLoader(self.val_dataset, batch_size=self.train_args.batch_size, num_workers=self.train_args.n_workers)
		self.test_loader = DataLoader(self.test_dataset, batch_size=self.train_args.batch_size, num_workers=self.train_args.n_workers)

		#TODO define the results path

	def loadGenerator(self):
		self.gen.load_state_dict(torch.load(self.eval_args.model_weights))
		pytorch_total_params = sum(p.numel() for p in self.gen.parameters())
		print("Generator params", pytorch_total_params)

	def loadDiscriminator(self):
		self.dis.load_state_dict(torch.load("./discriminators/noclosed/new_vanilla_discriminator.pth"))
		pytorch_total_params = sum(p.numel() for p in self.gen.parameters())
		print("Discriminator params", pytorch_total_params)

	def run(self):
		if (self.dis_path == None): #Model not pre-trained
			print("Pre-Training Discriminator")

	def randomiseTempo(self, generated):
		for seq in generated:
			for idx, value in enumerate(seq):
				if value >= 256 and value < 356:
					#						  never below 256		  never above 356
					seq[idx] = random.randint(max(256,seq[idx]-2), min(seq[idx]+2,356))
		return generated

	def randomiseVolume(self, generated):
		for seq in generated:
			for idx, value in enumerate(seq):
				if value >= 356:
					#						  never below 356		  never above 389
					seq[idx] = random.randint(max(356,seq[idx]-2), min(seq[idx]+2,389))
		return generated


	def generateSample(self, x):
		generated = self.gen.sample(x, self.parse_args.target_seq_length, one_hot=True)
		generated = generated.argmax(dim=2)
		#generated = torch.randint(388, (2, 128)).to(get_device()) test
		generated = self.randomiseTempo(generated)
		generated = self.randomiseVolume(generated)
		generated = torch.cat((x, generated), 1)  # batch size x 192
		return generated

	def discriminateSequence(self, seq):
		pred = self.dis(seq)
		pred = torch.squeeze(pred, 0)
		return pred

	def handlePredictions(self, pred, label, predictions, labels):
		round = torch.round(pred)
		for prediction in round:
			predictions.append(int(prediction.item()))
			labels.append(int(label[0].item())) #labels is a 2x1 tensor [0,0] or [1,1]


	def preTrainDiscriminator(self):
		self.dis.train()
		self.gen.eval()
		f1_score = []
		precision = []
		recall = []

		for epoch in range (0,self.train_args.pretrain_epochs):
			print("Epoch:", epoch)
			true_loss = []
			false_loss = []
			predictions = []
			labels = []
			target_names = ["artificial", "original"]

			for batch_num, batch in enumerate(self.train_loader):
				if (batch_num%50 == 0) : print("Iterating", batch_num)
				self.dis_opt.zero_grad()
				x = batch[0].to(get_device())
				tgt = batch[1].to(get_device())
				ran = random.randint(0,1)

				if (389 in x or 389 in tgt) : pass
				else:
					#case of tgt
					sample = torch.cat((x, tgt), 1)
					label = torch.ones([self.train_args.batch_size, 1], dtype=torch.float32).to(get_device())

					if (ran == 1): #case of generate
						sample = self.generateSample(x)
						sample = torch.cat((x, sample), 1)
						label = torch.zeros([self.train_args.batch_size, 1], dtype=torch.float32).to(get_device())

					pred = self.discriminateSequence(sample)
					self.handlePredictions(pred, label, predictions, labels)
					loss = self.dis_pretrain_criterion(pred, label)
					loss.backward()
					self.dis_opt.step()
					if (ran == 1) :
						false_loss.append(loss.detach().cpu().numpy().item())
					else :
						true_loss.append(loss.detach().cpu().numpy().item())

				if ((len(false_loss) > 0) & (len(true_loss) > 0)):
					if (batch_num % 100 == 0):
						print("false loss:", sum(false_loss) / len(false_loss))
						print("true loss:", sum(true_loss) / len(true_loss))
						report = classification_report(labels, predictions, target_names=target_names)
						print(report)

			report = classification_report(labels, predictions, target_names=target_names, output_dict=True)

			f1_score.append(report["weighted avg"]["f1-score"])
			precision.append(report["weighted avg"]["precision"])
			recall.append(report["weighted avg"]["recall"])

			fig = plt.figure()
			plt.plot(f1_score, label="f1-score")
			plt.plot(precision, label="precision")
			plt.plot(recall, label="recall")
			plt.legend()
			fig.savefig("./plots/new_vanilla_metrics.png")
			torch.save(self.dis.state_dict(), "./discriminators/new_vanilla_discriminator.pth")


	def trainBoth(self):
		gen_f1_score = []
		gen_precision = []
		gen_recall = []
		dis_f1_score = []
		dis_precision = []
		dis_recall = []

		for epoch in range(0, self.train_args.pretrain_epochs):
			true_loss = []
			false_loss = []
			gen_predictions = []
			gen_labels = []
			dis_predictions = []
			dis_labels = []
			target_names = ["artificial", "original"]
			print(epoch)
			for batch_num, batch in enumerate(self.train_loader):
				if (batch_num%10 == 0): print(batch_num)
				x = batch[0].to(get_device())
				tgt = batch[1].to(get_device())
				if (389 in x or 389 in tgt) : pass
				else:
					self.gen_opt.zero_grad()
					sample = self.generateSample(x)
					sample = torch.cat((x, sample), 1)
					# To calculate generator loss, generations need to have "true" label
					label_gen_1 = torch.ones([self.train_args.batch_size, 1], dtype=torch.float32).to(get_device())
					generated_dis = self.discriminateSequence(sample)
					gen_loss = self.dis_pretrain_criterion(generated_dis, label_gen_1)
					#This plot should be the opposite of the discriminator's one
					self.handlePredictions(generated_dis, label_gen_1, gen_predictions, gen_labels)
					gen_loss.backward(retain_graph=True)
					self.gen_opt.step()

					self.dis_opt.zero_grad()
					original = torch.cat((x, tgt), 1)
					label_true_1 = torch.ones([self.train_args.batch_size, 1], dtype=torch.float32).to(get_device())
					true_dis = self.discriminateSequence(original)
					true_dis_loss = self.dis_pretrain_criterion(true_dis, label_true_1)
					#We add the predictions of the true data to the plots handler
					self.handlePredictions(true_dis, label_gen_1, dis_predictions, dis_labels)

					label_gen_0 = torch.zeros([self.train_args.batch_size, 1], dtype=torch.float32).to(get_device())
					gen_dis_loss = self.dis_pretrain_criterion(generated_dis, label_gen_0)
					dis_total_loss = (true_dis_loss + gen_dis_loss) / 2
					# We add the predictions of the generated data to the plots handler
					self.handlePredictions(generated_dis, label_gen_0, dis_predictions, dis_labels)
					dis_total_loss.backward()
					self.dis_opt.step()

			report_gen = classification_report(gen_labels, gen_predictions, target_names=target_names, output_dict=True)
			report_dis = classification_report(dis_labels, dis_predictions, target_names=target_names, output_dict=True)

			gen_f1_score.append(report_gen["weighted avg"]["f1-score"])
			gen_precision.append(report_gen["weighted avg"]["precision"])
			gen_recall.append(report_gen["weighted avg"]["recall"])
			dis_f1_score.append(report_dis["weighted avg"]["f1-score"])
			dis_precision.append(report_dis["weighted avg"]["precision"])
			dis_recall.append(report_dis["weighted avg"]["recall"])

			fig_gen = plt.figure()
			plt.plot(gen_f1_score, label="f1-score")
			plt.plot(gen_precision, label="precision")
			plt.plot(gen_recall, label="recall")
			plt.legend()
			fig_gen.savefig("./plots/rand_generator_metrics.png")

			fig_dis = plt.figure()
			plt.plot(dis_f1_score, label="f1-score")
			plt.plot(dis_precision, label="precision")
			plt.plot(dis_recall, label="recall")
			plt.legend()
			fig_dis.savefig("./plots/rand_discriminator_metrics.png")

			if (epoch%5 == 0):
				torch.save(self.gen.state_dict(), "./generators/rand_gen_checkpoint_{}.pth".format(epoch))
				torch.save(self.dis.state_dict(), "./discriminators/rand_dis_checkpoint_{}.pth".format(epoch))