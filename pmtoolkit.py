import sys, os

import numpy as np
import h5py as h5
from scipy.signal import tukey
from scipy.interpolate import interp1d


# If you set the environment variable 'PYWIFES_DIR' then it will be found
NRWF_ENV = os.getenv('NRWAVEFORM_DIR')

# Otherwise, just look where we are right now, and go from there
if NRWF_ENV  == None:
    # Where are we located ?
    WF_DIR = os.path.dirname(__file__)
else:
    WF_DIR = NRWF_ENV

#useful units

cc = 299792458.0  # speed of light in m/s
GG = 6.67384e-11  # Newton in m^3 / (kg s^2)
Msun = 1.98855 * 10 ** 30  # solar mass in  kg
kg = 1. / Msun
metre = cc ** 2 / (GG * Msun)
secs = cc * metre
Mpc = 3.08568e+22  # Mpc in metres

# Functions for converting to geometric units
def m_sol_to_geo(mm):
	# convert from solar masses to geometric units
	return mm / kg * GG / cc ** 2

def dist_Mpc_to_geo( dist):
	# convert distance from Mpc to geometric units (i.e., metres)
	return dist * Mpc

def time_geo_to_s(time):
	# convert time from seconds to geometric units
	return time / cc

# Waveform models
def list_wf_labels():
	# list of waveform directories
	WF_AVAIL = [f for f in os.listdir(WF_DIR) if not f.startswith('.')]

	#list of waveform subdirectories that have the Rxx format
	WF_SUBDIR_AVAIL = []
	for this_dir in WF_AVAIL:
		this_path = os.path.join(WF_DIR,this_dir)
		this_subdir_avail =  [f for f in os.listdir(this_path) if 'R0' in f]
		for i in this_subdir_avail:
			WF_SUBDIR_AVAIL.append(os.path.join(this_dir,i))
	return WF_SUBDIR_AVAIL

# Parse the CORE models into something useable
class NRWaveform:
	def __init__(self, wf_label, loudness = 1):

		# load the CORE waveform
		this_waveform_fname = os.path.join(WF_DIR, wf_label, 'data.h5')
		print('opening {}...'.format(this_waveform_fname))
		f = h5.File(this_waveform_fname, 'r')

		rh22 = f['rh_22']
		keys = list(rh22.keys())

		# Extract the waveform computed at the largest mass coordinate from the file
		h22furthest = [key for key in keys if ('Rh_l2_m2' in key) and not ('Inf' in key)][-1]
		rh22data = rh22[h22furthest]
		hr_pl_msun = rh22data[:, 1] * 2 * 2 ** 0.5 # This scaling is an approximation
		hr_cr_msun = rh22data[:, 2] * 2 * 2 ** 0.5 # This scaling is an approximation

		# This is the time in geometric units
		time_msun = rh22data[:, 8]

		# The waveform starts wherever the amplitude is at a maximum
		postmerger_start_index= np.argmax((hr_pl_msun ** 2 + hr_cr_msun ** 2) ** 0.5)

		# rescaling time to non-geometric units
		self.tscale = time_geo_to_s(m_sol_to_geo(1))
		time_scaled = (time_msun - time_msun[postmerger_start_index]) * self.tscale
		self.tstartindex = np.argmax(time_scaled > - 0)

		# scale the amplitude, by default the loudness param is = 1
		self.hscale = loudness * 1.2 / np.max([np.max(np.abs(hr_pl_msun[postmerger_start_index:])),
			np.max(np.abs(hr_cr_msun[postmerger_start_index:]))])

		hre = hr_pl_msun * self.hscale
		him = hr_cr_msun * self.hscale


		# crop the waveforms to the start idx
		self.hrenew = hre[self.tstartindex:]
		self.himnew = him[self.tstartindex:]
		# crop the time array to match
		self.tnew = time_scaled[self.tstartindex:]

	def interpolate_wf_to_new_tarray(self, newtime, t_0, window = True, tukey_rolloff = 0.2):

		t_interp = self.tnew + t_0
		hplus_interp_func = interp1d(t_interp,
		                         self.hrenew,
		                         bounds_error=False, fill_value=0)

		hcross_interp_func = interp1d(t_interp,
		                         self.himnew,
		                         bounds_error=False, fill_value=0)

		# redefine tstartindex based on interpolated data
		tstartindex_new = np.argmax(newtime >= t_0 )
		tout = newtime[tstartindex_new:] # this is taken care of by the interpolation 

		self.hplus = np.zeros(newtime.shape)
		self.hcross = np.zeros(newtime.shape)

		# windowing
		if window == True:
			window_dt = tout[-1]-tout[0]
			tukey_rolloff_ms=  tukey_rolloff/1000    
			window = tukey(len(tout), 2 * tukey_rolloff_ms / window_dt) 


			self.hplus[tstartindex_new:]  = hplus_interp_func(tout) * window
			self.hcross[tstartindex_new:] = hcross_interp_func(tout) * window

		else:

			self.hplus[tstartindex_new:]  = hplus_interp_func(tout)
			self.hcross[tstartindex_new:] = hcross_interp_func(tout)


		return {'plus': self.hplus, 'cross': self.hcross}

class DampedSinWaveforms():
	def __init__(self, sample_rate):
		self.sample_rate = sample_rate

		def damped_sinusoid_td(self, duration, t_0, amplitude, damping_time, frequency, phase):

			self.time = np.linspace(t_0, duration, int((duration-t_0)*sample_rate))

			A = amplitude*np.exp(-time/damping_time)
			theta = 2*np.pi*frequency*time + phi

			self.hplus = A * np.sin(theta)
			self.hcross = A * np.cos(theta)

			return {'plus': self.hplus, 'cross': self.hcross}

		



