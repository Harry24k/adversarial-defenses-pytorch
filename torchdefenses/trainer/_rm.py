from datetime import datetime
import pickle
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._vis import init_plot, make_twin, plot_line, _to_array

class RecordManager(object):
    def __init__(self, keys):
        self.records = {}
        self._keys = keys
        
        # Mode 0: No Epoch, No Iter
        # Mode 1: Epoch Only, No Iter
        # Mode 2: Epoch, Iter 
        
        if keys[0] != 'Epoch':
            mode = 0
        else:
            if keys[1] != 'Iter':
                mode = 1
            else: 
                mode = 2
                
        self._mode = mode
                        
        self._record_len = 0
        self._text_len = 0
        
        self._start_time = datetime.now()
        self._progress_time = datetime.now()
        
        for key in self._keys:
            self.records[key] = []
            
        self._spinner = itertools.cycle(['-', '/', '|', '\\'])
    
    def progress(self):
        t = datetime.now() - self._progress_time
        print("Progress: "+ next(self._spinner) + " [" + str(t)+"/it]" + " "*20, end='\r')
        self._progress_time = datetime.now()
    
    def __repr__(self):
        return "RecordManager(keys=[%s])"%(", ".join(self._keys))
    
    def _head(self, values):
              
        lengths = []
        slack = 3
        for i, value in enumerate(values):
            length = max(len(str(value)), len(self._keys[i])) + slack
            if isinstance(value, float):
                length = max(len("%.4f"%(value)), len(self._keys[i])) + slack
            lengths.append(length)
        
        self._form = "".join(['{:<'+str(length)+'.'+str(length)+'}' for length in lengths])
#         if self._keys[0] == "Epoch":
#             self._mode = 1
#             self._form = ('{:<10.10}'*1 + '{:<15.15}'*(len(self._keys)-1))
#             if self._keys[1] == "Iter":
#                 self._form = ('{:<10.10}'*2 + '{:<15.15}'*(len(self._keys)-2))
#                 self._mode = 2
                
        text = self._form.format(*self._keys)
        self._text_len = len(text)
        print("-"*self._text_len)
        print(text)
        print("="*self._text_len)
    
    def add(self, values): 
        
        if len(values) != len(self._keys):
            raise ValueError('Values are NOT matched with Keys.')
            
        print(" "*50, end='\r')
        if self._record_len == 0:
            self._head(values)
            
        self._record_len += 1
        text_arr = []
        
        for i, value in enumerate(values):
            self.records[self._keys[i]].append(value)
            
            if isinstance(value, int):
                text_arr.append("%d"%(value))
            elif isinstance(value, float):
                text_arr.append("%.4f"%(value))
            else:
                raise ValueError('Only int of float is supported for a record item.')     
            
        print(self._form.format(*text_arr))
        print("-"*self._text_len)
       
    def summary(self): 
        print("="*self._text_len)
        
        if self._mode > 0:
            print("Total Epoch:", max(np.array(self.records["Epoch"])))
        else:
            print("Total Records:", self._record_len)
            
        print("Time Elapsed:", datetime.now() - self._start_time)
        
        if self._mode > 0:
            print("Min(epoch)/Max(epoch): ")
        else:
            print("Min(th)/Max(th): ")
            
        for i, key in enumerate(self._keys):
            history = np.array(self.records[key])
            
            if i < self._mode:
                continue
            
            if isinstance(self.records[key][0], (float, int)):
                argmin = history.argmin()
                argmax = history.argmax()
                
                if self._mode > 0:
                    pos_min = self.records["Epoch"][argmin]
                    pos_max = self.records["Epoch"][argmax]
                else:
                    pos_min = argmin+1
                    pos_max = argmax+1
                
                print("-"+key+": %.4f(%d)"%(history[argmin], pos_min)+
                      "/%.4f(%d)"%(history[argmax], pos_max))
                
        print("-"*self._text_len)
        
    def plot(self, x_key, y_keys, figsize=(6,6), title="", xlabel="", ylabel="",
             xlim=None, ylim=None, pad_ratio=0, tight=True, 
             linestyles=None, linewidths=None, colors=None, labels=None, alphas=None,
             ylabel_second="", ylim_second=None, legend=True, loc='best'):
                
        if self._mode > 0 and x_key == 'Epoch':
            data = self.to_dataframe().groupby('Epoch').tail(1)
        elif self._mode > 1 and x_key == 'Iter':
#             print("Warnings: This graph is an estimated graph based on Epoch/Iter.") 
            data = self.to_dataframe()
            data['Iter'] += (data['Epoch']-min(data['Epoch']))*max(data['Iter'])
        else: 
            data = self.to_dataframe()
            
        if not isinstance(y_keys, list):
            y_keys = [y_keys]
            
        # Check version and number of elements
        ver = 1 # e.g.["a"] or ["a", "b", "c"]
        length = 0
        y_keys_flat = []
        js = []
        lines2 = []
        labels2 = []
        
        for j, y_key in enumerate(y_keys):
            if isinstance(y_key, list):
                
                if len(y_keys) > 2:
                    raise RuntimeError("Axes can have the maximum value as 2.")
                    
                for y in y_key:
                    y_keys_flat.append(y)
                    length += 1
                    js.append(j)
                    
                ver = 2 # e.g. [["a", "b"], "c"] or [["a", "b"], ["c", "d"]]

            else:
                y_keys_flat.append(y_key)
                length += 1
                js.append(j)

        x = data[x_key]
        inputs = [data[y_key] for y_key in y_keys_flat]
            
        # Draw plots
        ax = init_plot(ax=None, figsize=figsize, title=title, xlabel=xlabel, ylabel=ylabel,
                       xlim=xlim, ylim=ylim, pad_ratio=pad_ratio, tight=tight)
        
        linestyles, linewidths, colors, labels, alphas = _to_array([linestyles, linewidths, colors, labels, alphas], length)
        
        i = 0    
        if ver == 1:
            for input in inputs:
                plot_line(ax, x, input, linestyle=linestyles[i], linewidth=linewidths[i],
                          color=colors[i], label=labels[i], alpha=alphas[i])
                i += 1

        elif ver == 2:
            ax2 = make_twin(ax=ax, ylabel=ylabel_second, ylim=ylim_second)
            axes = [ax, ax2]
            for j, input in enumerate(inputs):
                plot_line(axes[js[j]], x, input, linestyle=linestyles[i], linewidth=linewidths[i],
                          color=colors[i], label=labels[i], alpha=alphas[i])
                i += 1
                
            lines2, labels2 = ax2.get_legend_handles_labels()
                
        else:
            raise RuntimeError("Unreadable inputs")
            
        if legend:
            lines, labels = ax.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc=loc)
            
        plt.show()
            
    def to_dataframe(self, keys=None):
        if keys == None:
            keys = self._keys
            
        data = pd.DataFrame(columns=[*keys])
        
        for key in keys:
            data[key] = np.array(self.records[key])
        
        return data
        
    def to_csv(self, path, verbose=True):
        data = self.to_dataframe()            
        data.to_csv(path, mode="w", index=False)
        if verbose:
            print("...Saved as csv to", path, "!")
        
    def save(self, path):
        with open(path, "wb") as fp:   #Pickling
            pickle.dump(self.records, fp)
        print("...Saved as pickle to", path, "!")
        
    def load(self, path):
        with open(path, "rb") as fp:   # Unpickling
            return pickle.load(fp)