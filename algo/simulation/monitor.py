import json, os, subprocess, sys, tkinter as tk
from tkinter import ttk

TELEMETRY = '/tmp/algo_simulator_telemetry.json'

class Monitor:
    def __init__(self):
        self.root = tk.Tk(); self.root.title('Algorithm run monitor'); self.root.geometry('760x560')
        self.status = tk.StringVar(value='Starting simulator...')
        ttk.Label(self.root, textvariable=self.status, font=('Helvetica', 15, 'bold')).pack(anchor='w', padx=12, pady=10)
        self.metrics = ttk.Label(self.root); self.metrics.pack(anchor='w', padx=12)
        cols = ('leg','target','planned','moved','status'); self.table = ttk.Treeview(self.root, columns=cols, show='headings', height=10)
        for col, title in zip(cols, ('Leg','Image','Planned cm','Moved cm','Status')):
            self.table.heading(col, text=title); self.table.column(col, width=130, anchor='center')
        self.table.pack(fill='x', padx=12, pady=10)
        ttk.Label(self.root, text='Event history', font=('Helvetica', 11, 'bold')).pack(anchor='w', padx=12)
        self.events = tk.Listbox(self.root, height=12); self.events.pack(fill='both', expand=True, padx=12, pady=4)
        self.last_state = self.last_action = None; self.root.protocol('WM_DELETE_WINDOW', self.close); self.root.after(100, self.refresh)

    def refresh(self):
        try:
            with open(TELEMETRY) as f: state = json.load(f)
        except (OSError, json.JSONDecodeError):
            self.root.after(100, self.refresh); return
        elapsed, index, step = state['elapsed_seconds'], state['animation_index'], state['step_cm']
        moved = max(0, min(index + 1, state['node_count']))
        self.status.set(f"LIVE | phase: {state['task_state']} | algorithm: {state.get('algorithm', 'unknown')} | node: {moved}/{state['node_count']}")
        self.metrics.config(text=f"Moved: {moved*step:.1f} cm    Planned: {state['node_count']*step:.1f} cm    Images: {state['recognised']}/{state['target_images']}")
        if state['task_state'] != self.last_state:
            self.events.insert(tk.END, f"{elapsed:7.2f}s  PHASE -> {state['task_state']}"); self.last_state = state['task_state']
        algorithm = state.get('algorithm', 'unknown')
        if not hasattr(self, 'last_algorithm') or algorithm != self.last_algorithm:
            self.events.insert(tk.END, f"{elapsed:7.2f}s  ALGORITHM -> {algorithm}"); self.last_algorithm = algorithm
        actions = state.get('actions', []); action = actions[index] if 0 <= index < len(actions) else None
        if action != self.last_action and action:
            self.events.insert(tk.END, f"{elapsed:7.2f}s  CONTROL -> {action['gear']} / {action['steering']}"); self.last_action = action
        self.table.delete(*self.table.get_children()); previous = -1
        for number, end in enumerate(state.get('leg_ends', []), 1):
            count, current = end - previous, max(0, min(index - previous, end - previous))
            self.table.insert('', 'end', values=(number, f'Image {number}', f'{count*step:.1f}', f'{current*step:.1f}', 'DONE' if index >= end else 'PENDING')); previous = end
        self.root.after(100, self.refresh)

    def close(self): self.root.destroy()

def main():
    try: os.remove(TELEMETRY)
    except OSError: pass
    simulator = subprocess.Popen([sys.executable, '-m', 'algo.simulation.simulator'])
    Monitor().root.mainloop()
    if simulator.poll() is None: simulator.terminate()

if __name__ == '__main__': main()
