import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display,clear_output
from scipy.optimize import curve_fit

def fit():
	wl = widgets.Layout(width='80%',height='24pt')
	wbl = widgets.Layout(width='2in',height='0.25in')
	ws = {'description_width':'initial'}

	out = widgets.Output()

	filename = widgets.Textarea(value='',placeholder='Enter file name (.csv)',description="Data file name",layout=wl,style=ws)
	label_y = widgets.Textarea(value='Signal (A.U.)',placeholder='Enter Y axis label',description="Y axis label",layout=wl,style=ws)
	label_x = widgets.Textarea(value='Independent Var (A.U.)',placeholder='Enter X axis label',description="X axis label",layout=wl,style=ws)

	float_xmin = widgets.FloatText(value=0.,description='X Range (min)',style=ws)
	float_xmax = widgets.FloatText(value=0.,description='X Range (max)',style=ws)
	dropdown_fxn = widgets.Dropdown(value='Single Exponential', options=['Linear','Quadratic','Single Exponential','Double Exponential'],description='Fitting Function',style=ws)

	button_fit = widgets.Button(description='Fit',layout=wbl,style=ws)
	button_plot = widgets.Button(description='Show Data',layout=wbl,style=ws)

	vbox = widgets.VBox([float_xmin,float_xmax,dropdown_fxn])
	hbox = widgets.HBox([button_plot,button_fit,])
	total = widgets.VBox([filename,label_x,label_y,vbox,hbox])

	def show_ui():
		with out:
			out.clear_output()
			display(total)

	def fxn_exp1(x,A1,k1,B):
		return A1 * np.exp(-k1*x) + B
	def fxn_exp2(x,A1,k1,A2,k2,B):
		if k1>=k2:
			return x*np.nan
		return A1 * np.exp(-k1*x) + A2 * np.exp(-k2*x) + B
	def fxn_linear(x,A,B):
		return A*x+B
	def fxn_quadratic(x,A,B,C):
		return A*x**2.+B*x+C

	def get_fxn():
		if dropdown_fxn.value == 'Linear':
			fxn = fxn_linear
		elif dropdown_fxn.value == 'Quadratic':
			fxn = fxn_quadratic
		elif dropdown_fxn.value == 'Single Exponential':
			fxn = fxn_exp1
		elif dropdown_fxn.value == 'Double Exponential':
			fxn = fxn_exp2
		return fxn

	def make_plot(x,y,keep,theta=None):
		fxn = get_fxn()
		
		if not theta is None:
			fig,ax = plt.subplots(2,sharex=True,figsize=(4,4),height_ratios=[4, 1],)
			ax[0].plot(x,y,color='black')
			ax[1].axhline(y=0,color='black')

			xx = x[keep]
			yy = y[keep]
			fx = np.linspace(xx.min(),xx.max(),1000)
			fy = fxn(fx-fx[0],*theta)
			residual = fxn(xx-xx[0],*theta) - yy

			ax[0].plot(fx,fy,color='tab:red')
			ax[1].plot(xx,residual,color='tab:red')
			delta = np.max(np.abs(residual))*1.05
			ax[1].set_ylim(-delta,delta)
			ax[1].set_ylabel('Residual')
			ax[1].set_xlabel(label_x.value)
			ax[0].set_xlim(x.min(),x.max())
			ax[0].set_ylabel(label_y.value)
		else:
			fig,ax = plt.subplots(1,sharex=True,figsize=(4,3),)
			ax.plot(x,y,color='black')
			ax.set_xlim(x.min(),x.max())
			ax.set_ylabel(label_y.value)

		plt.tight_layout()
		return fig,ax

	def load():
		try:
			x,y = np.loadtxt(filename.value,delimiter=',').T
		except Exception as e:
			# print(e)
			print(f'ERROR: Could not load "{filename.value}". Check file')
			return None,None
		return x,y

	def get_keep(x,y):
		# if float_xmin.value >= float_xmax.value:
		# 	float_xmin.value = x.min()
		# 	float_xmax.value = x.max()

		keep = np.bitwise_and(x>=float_xmin.value,x<=float_xmax.value)
		xx = x[keep]
		yy = y[keep]
		return xx,yy,keep

	def guess_theta():
		x,y = load()
		if x is None:
			return None
		xx,yy,keep = get_keep(x,y)
		if keep.sum() == 0:
			print('X-axis range is empty')
			return None
		
		A = yy.max()-yy.min() if yy.argmax() < yy.argmin() else yy.min()-yy.max()
		k = 10./(xx.max()-xx.min())
		B = yy.mean()
		
		if dropdown_fxn.value == 'Linear':
			return np.polyfit(xx,yy,1)
		elif dropdown_fxn.value == 'Quadratic':
			return np.polyfit(xx,yy,2)
		elif dropdown_fxn.value == 'Single Exponential':
			return np.array((A,k,B))
		elif dropdown_fxn.value == 'Double Exponential':
			return np.array((A/2.,k/2.,A/2.,k*2.,B))

	def click_fit(b):
		with out:
			show_ui()

			try:
				x,y = np.loadtxt(filename.value,delimiter=',').T
			except Exception as e:
				# print(e)
				print(f'ERROR: Could not load "{filename.value}". Check file')
				return

			theta = guess_theta()
			if 	theta is None:
				print('Guessing failed')
				return

			xx,yy,keep = get_keep(x,y)
			if keep.sum() == 0:
				print('X-axis range is empty')
				return

			try:
				fxn = get_fxn()
				for iters in range(2):
					theta,cov = curve_fit(fxn,xx-xx[0],yy,p0=theta,maxfev=10000,)
				sig = np.sqrt(np.diag(cov))
				model = fxn(xx-xx[0],*theta)

				ss_res = np.sum((yy-model)**2.)
				ss_tot = np.sum((yy - np.mean(yy))**2.)
				r_squared = 1.-(ss_res/ss_tot)

				if dropdown_fxn.value == 'Linear':
					params = ['m','b']
				elif dropdown_fxn.value == 'Quadratic':
					params = ['A','B','C']
				elif dropdown_fxn.value == 'Single Exponential':
					params = ['A','k','B']
				elif dropdown_fxn.value == 'Double Exponential':
					params = ['A1','k1','A2','k2','B']

				with open('fitting_results.txt','w') as f:
					f.write('Fitting Results:\n')
					for i in range(len(params)):
						f.write(f'{params[i]} = {theta[i]:.6f}+/-{sig[i]:.6f}\n')
					f.write(f'R^2 = {r_squared:.6f}\n')
				with open('fitting_results.txt','r') as f:
					[print(line) for line in f]

				fig,ax = make_plot(x,y,keep,theta)
				plt.savefig('fitted_data.pdf')
				plt.savefig('fitted_data.png')
				plt.show()
				plt.close()

				with open('fitting_fit.txt','w') as f:
					for i in range(xx.size):
						f.write(f'{xx[i]},{yy[i]},{model[i]},{yy[i]-model[i]}\n')

			except Exception as e:
				print(e)
				print(f'ERORR: Fitting Failed. Try a better initial guess?')

	def click_plot(b):
		with out:
			show_ui()
			x,y = load()
			if x is None:
				return None
			
			float_xmin.value = x.min()
			float_xmax.value = x.max()
			xx,yy,keep = get_keep(x,y)

			fig,ax = make_plot(x,y,keep,None,)
			plt.show()
			plt.close()

	button_fit.on_click(click_fit)
	button_plot.on_click(click_plot)
	show_ui()
	display(out)
