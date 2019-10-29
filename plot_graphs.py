import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib import gridspec
import seaborn as sns
import os

CC_color = 'red'
CS_color = 'green'
bin_size = 10
input_cond = 2
start = 0

if input_cond == 1:
	# 200 pA (500 to 600) and 400 pA (1500 to 1600)
	data = sio.loadmat('MATLAB/200_400_pA.mat')
	input_dist = np.random.normal(10/100, math.sqrt(5)/100, 2400) 
	input_dist[500:600] += 2
	input_dist[1500:1600] += 4
	inputs = [[0,2400],[500, 1000], [1500, 2000]]
	folder = '200_400_imgs'

else:
	# noise ~N(200,200)
	input_dist = np.random.normal(200/100, 200/100, 2400)
	inputs = [[0,2400]]
	data = sio.loadmat('MATLAB/200_noise.mat')
	folder = '200_noise_imgs'

if not os.path.exists(folder):
	os.makedirs(folder)

def get_mean(f_f):
	return np.mean(f_f, 1)

def get_std(f_f):
	return np.std(f_f, 1)

def get_var(f_f):
	return np.var(f_f, 1)

network_ctrl = data['f_rates_ctrl']
network_stim = data['f_rates_stim']

network_ctrl_mean = get_mean(network_ctrl)
network_stim_mean = get_mean(network_stim)

proj_ctrl_f = network_ctrl[:,:270]
proj_stim_f = network_stim[:,:270]

CC_ctrl_f = data['CC']['ctrl_f_ws'][0][0]
CS_ctrl_f = data['CS']['ctrl_f_ws'][0][0]
VIP_ctrl_f = data['VIP']['ctrl_f_ws'][0][0]
SST_ctrl_f = data['SST']['ctrl_f_ws'][0][0]
PV_ctrl_f = data['PV']['ctrl_f_ws'][0][0]

CC_stim_f = data['CC']['stim_f_ws'][0][0]
CS_stim_f = data['CS']['stim_f_ws'][0][0]
VIP_stim_f = data['VIP']['stim_f_ws'][0][0]
SST_stim_f = data['SST']['stim_f_ws'][0][0]
PV_stim_f = data['PV']['stim_f_ws'][0][0]

CC_ctrl_f_mean = get_mean(CC_ctrl_f)
CS_ctrl_f_mean = get_mean(CS_ctrl_f)
VIP_ctrl_f_mean = get_mean(VIP_ctrl_f)
SST_ctrl_f_mean = get_mean(SST_ctrl_f)
PV_ctrl_f_mean = get_mean(PV_ctrl_f)

CC_stim_f_mean = get_mean(CC_stim_f)
CS_stim_f_mean = get_mean(CS_stim_f)
VIP_stim_f_mean = get_mean(VIP_stim_f)
SST_stim_f_mean = get_mean(SST_stim_f)
PV_stim_f_mean = get_mean(PV_stim_f)

def plot_optimization():
	ax1 = plt.subplot(111)
	w0_eig = data['w0_eig']
	X0 = [w0_eig.real for x in w0_eig]
	Y0 = [w0_eig.imag for x in w0_eig]
	ax1.scatter(X0,Y0, color='gray',alpha=0.5, marker='x',s=5, label='Before Optimization')

	wsoc_eig = data['wsoc_eig']
	Xsoc = [wsoc_eig.real for x in wsoc_eig]
	Ysoc = [wsoc_eig.imag for x in wsoc_eig]
	ax1.scatter(Xsoc,Ysoc, color='black', marker='x',s=5, label='After Optimization')
	# ax1.grid(alpha=0.1)

	ax1.set_ylabel('Imaginary axis', fontsize=15)
	ax1.set_xlabel('Real axis', fontsize=15)
	ax1.legend(fontsize=15)
	ax1.vlines(1.80,-10,10,linestyles='dashed',color='r',alpha=0.5, linewidth=3)
	ax1.set_xticklabels(list(range(-10,10,2)), fontsize = 15)
	ax1.set_yticklabels(list(range(-10,10,2)), fontsize = 15)
	ax1.set_xlim(-10,10)
	ax1.set_ylim(-10,10)
	plt.show()
	plt.savefig(folder+'/W_eig.png')

	SA_values = data['SA_values'][0]
	ax1.plot(SA_values, linewidth=3)
	ax1.set_xlabel('Number of iterations', fontsize=15)
	ax1.set_ylabel('Spectral Abscissa', fontsize=15)
	ax1.set_xticklabels([-100, 0, 100,200,300,400,500,600], fontsize = 15)
	ax1.set_yticklabels(list(range(0,20,2)), fontsize = 15)
	ax1.hlines(1.75,0,650,linestyles='dashed',color='r',alpha=0.5, linewidth=3)
	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.yaxis.set_ticks_position('left')
	ax1.xaxis.set_ticks_position('bottom')
	plt.show()
	plt.savefig(folder+'/SA_values.png')

	def extract_subset(matrix, ratio):
		e_size = int(270*ratio)
		i_size = 270+int(30*ratio)
		CC_size = int(180*ratio)
		CS_size = 180+int(90*ratio)
		VIP_size = 270+int(10*ratio)
		SST_size = 280+int(10*ratio)
		PV_size = 290+int(10*ratio)
		
		CC_matrix = np.concatenate((matrix[:e_size,:CC_size],matrix[270:i_size,:CC_size]),axis=0)
		CS_matrix = np.concatenate((matrix[:e_size,180:CS_size],matrix[270:i_size,180:CS_size]),axis=0)
		VIP_matrix = np.concatenate((matrix[:e_size,270:VIP_size],matrix[270:i_size,270:VIP_size]),axis=0)
		SST_matrix = np.concatenate((matrix[:e_size,280:SST_size],matrix[270:i_size,280:SST_size]),axis=0)
		PV_matrix = np.concatenate((matrix[:e_size,290:PV_size],matrix[270:i_size,290:PV_size]),axis=0)
		combined = np.concatenate((CC_matrix,CS_matrix,VIP_matrix,SST_matrix,PV_matrix), axis=1)
		return combined

	w0 = extract_subset(data['W0'],0.5)
	wsoc = extract_subset(data['Wsoc'], 0.5)

	sns.heatmap(w0, square=True, xticklabels=False, yticklabels=False, cmap="RdBu_r", vmin=-3, vmax=3)
	plt.savefig(folder+'/w0.png')

	sns.heatmap(wsoc, square=True, xticklabels=False, yticklabels=False, cmap="RdBu_r", vmin=-3, vmax=3)
	plt.savefig(folder+'/wsoc.png')

# plot_optimization()

def plot_all():
	fig = plt.figure(figsize=(40,20))
	gs = gridspec.GridSpec(9, 2)
	ax1 = plt.subplot(gs[0:4,0])
	ax2 = plt.subplot(gs[4:8,0], sharex=ax1)
	ax3 = plt.subplot(gs[8,0], sharex=ax1)

	ax4 = plt.subplot(gs[0:4,1], sharey=ax1, sharex=ax1)
	ax5 = plt.subplot(gs[4:8,1],sharex=ax4, sharey=ax2)
	ax6 = plt.subplot(gs[8,1],sharex=ax4, sharey=ax3)

	x = np.arange(2400)
	y = input_dist
	ax3.plot(x[start:],y[start:],color='black')
	ax3.get_yaxis().set_visible(False)
	ax3.set_ylim(0,10)
	ax3.axis('off')

	ax6.plot(x[start:],y[start:],color='black')
	ax6.get_yaxis().set_visible(False)
	ax6.set_ylim(0,10)
	ax6.axis('off')

	for i in range(len(CC_ctrl_f[0])):
		ax1.plot(CC_ctrl_f[:,i][start:], alpha=0.5)

	for i in range(len(CC_stim_f[0])):
		ax4.plot(CC_stim_f[:,i][start:], alpha=0.5)
		
	for i in range(len(CS_ctrl_f[0])):
		ax2.plot(CS_ctrl_f[:,i][start:], alpha=0.5)
		
	for i in range(len(CS_ctrl_f[0])):
		ax5.plot(CS_stim_f[:,i][start:], alpha=0.5)
		
	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.set_title('CC Control')

	ax2.spines['right'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax2.set_title('CS control')
	
	ax4.spines['right'].set_visible(False)
	ax4.spines['top'].set_visible(False)
	ax4.set_title('CC Stim')
	
	ax5.spines['right'].set_visible(False)
	ax5.spines['top'].set_visible(False)
	ax5.set_title('CS Stim')

	plt.savefig(folder+'/all.png')

def plot_CC_CS_firing_figs():
	fig = plt.figure(figsize=(20,10))
	gs = gridspec.GridSpec(5, 2)
	ax1 = plt.subplot(gs[0:4,0])
	ax2 = plt.subplot(gs[4, 0])
	ax3 = plt.subplot(gs[0:4, 1], sharey=ax1)
	ax4 = plt.subplot(gs[4, 1], sharey=ax2)

	ax1.plot(CC_ctrl_f_mean,label='CC', color=CC_color, linewidth=5)
	ax1.plot(CS_ctrl_f_mean,label='CS', color=CS_color, linewidth=5)
	ax1.set_xlabel('Time (arbitrary unit)', fontsize=15)
	ax1.set_ylabel('Firing Rates (Hz)', fontsize=15)
	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.set_title('Control', fontsize=15)
	ax1.legend(fontsize=15)

	x = np.arange(2400)
	y = input_dist
	ax2.plot(x,y,color='black')
	ax2.get_yaxis().set_visible(False)
	ax2.set_ylim(min(y)-1,max(y)+1)
	ax2.axis('off')

	ax3.plot(CC_stim_f_mean,label='CC', color=CC_color, linewidth=5)
	ax3.plot(CS_stim_f_mean,label='CS', color=CS_color, linewidth=5)
	ax3.set_xlabel('Time (arbitrary unit)', fontsize=15)
	ax3.set_ylabel('Firing Rates (Hz)', fontsize=15)
	ax3.spines['right'].set_visible(False)
	ax3.spines['top'].set_visible(False)
	ax3.set_title('ACh Stimulation', fontsize=15)
	ax3.legend(fontsize=15)

	ax4.plot(x,y,color='black')
	ax4.get_yaxis().set_visible(False)
	ax4.set_ylim(min(y)-1,max(y)+1)
	ax4.axis('off')

	plt.savefig(folder+'/CC_CS_mean.png')

def plot_ctrl_stim_figs():
	fig = plt.figure(figsize=(20,10))
	gs = gridspec.GridSpec(5, 2)
	ax1 = plt.subplot(gs[0:4,0])
	ax2 = plt.subplot(gs[4, 0])
	ax3 = plt.subplot(gs[0:4, 1], sharey=ax1)
	ax4 = plt.subplot(gs[4, 1], sharey=ax2)

	ax1.plot(CC_ctrl_f_mean,label='Control', color=CC_color, alpha=0.5, linestyle='dotted',linewidth=5)
	ax1.plot(CC_stim_f_mean,label='ACh Stimulation', color=CC_color, linewidth=5)
	ax1.set_xlabel('Time (arbitrary unit)', fontsize=15)
	ax1.set_ylabel('Firing Rates (Hz)', fontsize=15)
	ax1.spines['right'].set_visible(False)
	ax1.spines['top'].set_visible(False)
	ax1.set_title('CC', fontsize=15)
	ax1.legend(fontsize=15)

	x = np.arange(2400)
	y = input_dist
	ax2.plot(x,y,color='black')
	ax2.get_yaxis().set_visible(False)
	ax2.set_ylim(min(y)-1,max(y)+1)
	ax2.axis('off')

	ax3.plot(CS_ctrl_f_mean,label='Control', color=CS_color, alpha=0.5, linestyle='dotted',linewidth=5)
	ax3.plot(CS_stim_f_mean,label='ACh Stimulation', color=CS_color, linewidth=5)
	ax3.set_xlabel('Time (arbitrary unit)', fontsize=15)
	ax3.set_ylabel('Firing Rates (Hz)', fontsize=15)
	ax3.spines['right'].set_visible(False)
	ax3.spines['top'].set_visible(False)
	ax3.set_title('CS', fontsize=15)
	ax3.legend(fontsize=15)

	ax4.plot(x,y,color='black')
	ax4.get_yaxis().set_visible(False)
	ax4.set_ylim(min(y)-1,max(y)+1)
	ax4.axis('off')

	plt.savefig(folder+'/CC_CS_mean_2.png')

def get_bin_idx(i, bin_size, min_d):
	return int(i*bin_size)-min_d

def generate_discrete_bins(rates, bin_size):
	min_d, max_d = int(min(rates)*bin_size), int(max(rates)*bin_size)
	bins = {}
	for rate in rates:
		idx = get_bin_idx(rate, bin_size, min_d)
		if idx not in bins:
			bins[idx] = 0
		bins[idx] += 1
	for idx in bins:
		bins[idx] /= len(rates)
	# x_axis = np.arange(min_d-1, max_d+1, 1)
	# plt.bar(x_axis, bins)
	# plt.show()
	return min_d, bins

def get_entropies(rates):
	entropies = []
	for t in range(len(rates)):
		pop_corr, pop_probs = generate_discrete_bins(rates[t],bin_size)
		entropy = -np.sum(list(map(lambda x:0.5*pop_probs[x]*math.log(pop_probs[x],2), pop_probs)))
		entropies.append(entropy)
	return np.array(entropies)

def get_r_prob(rate, bin_size, r_corr, r_probs):
	idx = get_bin_idx(rate, bin_size, r_corr)
	return r_probs[idx]

def get_information(ctrl_f, stim_f):
	all_f = np.concatenate((ctrl_f, stim_f),1)
	r_corr, r_probs = generate_discrete_bins(all_f.flatten(),bin_size)
	def helper(rates):
		info = []
		for t in range(len(rates)):
			rate = rates[t]
			pop_corr, pop_probs = generate_discrete_bins(rate,bin_size)
			curr_info = 0
			for n in rate:
				pop_idx = get_bin_idx(n, bin_size, pop_corr)
				pop_prob = pop_probs[pop_idx]
				r_idx = get_bin_idx(n, bin_size, r_corr)
				r_prob = r_probs[r_idx]
				curr_info += 0.5*pop_prob*math.log((pop_prob/r_prob),2)
			info.append(curr_info)
		return np.array(info)
	ctrl_info = helper(ctrl_f)
	stim_info = helper(stim_f)
	return [ctrl_info, stim_info]

def plot_H_I_maxH_eff():
	def plot_helper(CC_ctrl, CC_stim, CS_ctrl, CS_stim, ylabel, linewidth, start, filename):
		fig = plt.figure(figsize=(40,10))
		gs = gridspec.GridSpec(5, 2)
		ax1 = plt.subplot(gs[0:4, 0])
		ax2 = plt.subplot(gs[4, 0])
		ax3 = plt.subplot(gs[0:4, 1], sharey=ax1)
		ax4 = plt.subplot(gs[4, 1], sharey=ax2)

		ax1.plot(CC_ctrl[start:], label='Control', color=CC_color, alpha=0.5, linewidth=linewidth, linestyle='dotted')
		ax1.plot(CC_stim[start:], label='Stim', color=CC_color, linewidth=linewidth)
		ax1.set_xlabel('Time (arbitrary unit)', fontsize=15)
		ax1.set_ylabel(ylabel, fontsize=15)
		ax1.spines['right'].set_visible(False)
		ax1.spines['top'].set_visible(False)
		ax1.set_title('CC', fontsize=15)
		ax1.legend(fontsize=15)

		ax3.plot(CS_ctrl[start:], label='Control', color=CS_color, alpha=0.5, linewidth=linewidth, linestyle='dotted')
		ax3.plot(CS_stim[start:], label='Stim', color=CS_color, linewidth=linewidth)
		ax3.set_xlabel('Time (arbitrary unit)', fontsize=15)
		ax3.set_ylabel(ylabel, fontsize=15)
		ax3.spines['right'].set_visible(False)
		ax3.spines['top'].set_visible(False)
		ax3.set_title('CS', fontsize=15)
		ax3.legend(fontsize=15)

		x = np.arange(2400)
		y = input_dist
		ax2.plot(x[start:],y[start:],color='black')
		ax2.get_yaxis().set_visible(False)
		ax2.set_ylim(0,10)
		ax2.axis('off')

		x = np.arange(2400)
		y = input_dist
		ax4.plot(x[start:],y[start:],color='black')
		ax4.get_yaxis().set_visible(False)
		ax4.set_ylim(0,10)
		ax4.axis('off')

		plt.savefig(folder+'/'+filename)

	linewidth = 5
	def helper1(CC_ctrl_f, CC_stim_f, CS_ctrl_f, CS_stim_f):
		CC_ctrl_entropies = get_entropies(CC_ctrl_f)
		CC_stim_entropies = get_entropies(CC_stim_f)
		CS_ctrl_entropies = get_entropies(CS_ctrl_f)
		CS_stim_entropies = get_entropies(CS_stim_f)

		ylabel = 'Entropy Given Stimulus (bits)'
		filename = 'CC_CS_H_S.png'
		plot_helper(CC_ctrl_entropies, CC_stim_entropies, CS_ctrl_entropies, CS_stim_entropies, ylabel, linewidth, start, filename)

		CC_ctrl_info, CC_stim_info = get_information(CC_ctrl_f, CC_stim_f)
		CS_ctrl_info, CS_stim_info	 = get_information(CS_ctrl_f, CS_stim_f)

		ylabel = 'Information (bits)'	
		filename = 'CC_CS_info.png'
		plot_helper(CC_ctrl_info, CC_stim_info, CS_ctrl_info, CS_stim_info, ylabel, linewidth, start, filename)

		CC_ctrl_maxH = CC_ctrl_info+CC_ctrl_entropies
		CC_stim_maxH = CC_stim_info+CC_stim_entropies
		CS_ctrl_maxH = CS_ctrl_info+CS_ctrl_entropies
		CS_stim_maxH = CS_stim_info+CS_stim_entropies

		ylabel = 'Entropy (bits)'	
		filename = 'CC_CS_H.png'
		plot_helper(CC_ctrl_maxH, CC_stim_maxH, CS_ctrl_maxH, CS_stim_maxH, ylabel, linewidth, start, filename)

		CC_ctrl_efficiencies = CC_ctrl_info/CC_ctrl_maxH
		CC_stim_efficiencies = CC_stim_info/CC_stim_maxH
		CS_ctrl_efficiencies = CS_ctrl_info/CS_ctrl_maxH
		CS_stim_efficiencies = CS_stim_info/CS_stim_maxH

		ylabel = 'Efficiency'	
		filename = 'CC_CS_E.png'
		plot_helper(CC_ctrl_efficiencies, CC_stim_efficiencies, CS_ctrl_efficiencies, CS_stim_efficiencies, ylabel, linewidth, start, filename)

	helper1(CC_ctrl_f, CC_stim_f, CS_ctrl_f, CS_stim_f)

def plot_energies_and_SNR():
	def plot_helper(ctrl, stim, title, color, linewidth, filename, start, end):
		fig = plt.figure(figsize=(20,10))
		gs = gridspec.GridSpec(5, 2)
		ax1 = plt.subplot(gs[0:5, 0])
		ax3 = plt.subplot(gs[0:5, 1])

		max_e = max(max(ctrl), max(stim))
		x_axis = np.arange(len(ctrl))
		ax1.bar(x_axis, ctrl, label='Control', color=color, linewidth=linewidth)
		ax1.set_xlabel('Neurons', fontsize=15)
		ax1.set_ylabel('Power Spectral Density', fontsize=15)
		ax1.spines['right'].set_visible(False)
		ax1.spines['top'].set_visible(False)
		ax1.set_title('{} Control ({}ms to {}ms)'.format(title, start, end), fontsize=15)
		ax1.set_ylim(0, max_e)

		ax3.bar(x_axis, stim, label='ACh Stimulation', color=color, linewidth=linewidth)
		ax3.set_xlabel('Neurons', fontsize=15)
		ax3.set_ylabel('Power Spectral Density', fontsize=15)
		ax3.spines['right'].set_visible(False)
		ax3.spines['top'].set_visible(False)
		ax3.set_title('{} Stim ({}ms to {}ms)'.format(title, start, end), fontsize=15)
		ax3.set_ylim(0, max_e)

		plt.savefig(folder+'/'+filename)

	def helper(ctrl_f, stim_f, title, color, linewidth, start, end):
		filename = '{}_energies_{}_{}.png'.format(title, start, end)
		ctrl_energies = []
		stim_energies = []
		for i in range(len(ctrl_f[0])):
			ctrl_ft = np.fft.fft(ctrl_f[:,i])
			ctrl_energy = np.sum(np.abs(ctrl_ft)**2)
			ctrl_energies.append(ctrl_energy)

			stim_ft = np.fft.fft(stim_f[:,i])
			stim_energy = np.sum(np.abs(stim_ft)**2)
			stim_energies.append(stim_energy)
		
		ctrl_mean = np.mean(ctrl_f[start:end],1)
		stim_mean = np.mean(stim_f[start:end],1)

		ctrl_ft = np.fft.fft(ctrl_mean)
		stim_ft = np.fft.fft(stim_mean)
		ctrl_signal = np.sum((np.abs(ctrl_ft)**2))
		stim_signal = np.sum((np.abs(stim_ft)**2))

		ctrl_noise = np.mean(ctrl_energies)
		stim_noise = np.mean(stim_energies)
		print('{} Ctrl SNR: {}\n{} Stim SNR: {}\n'.format(title, ctrl_signal/ctrl_noise, title, stim_signal/stim_noise))

		plot_helper(ctrl_energies, stim_energies, title, color, linewidth, filename, start, end)

	for input_slot in inputs:
		start, end = input_slot
		print('From {}ms to {}ms\n'.format(start, end))
		helper(CC_ctrl_f, CC_stim_f, 'CC', 'red', 5, start, end)
		helper(CS_ctrl_f, CS_stim_f, 'CS', 'green', 5, start, end)

plot_all()
plot_CC_CS_firing_figs()
plot_ctrl_stim_figs()
plot_H_I_maxH_eff()
plot_energies_and_SNR()
