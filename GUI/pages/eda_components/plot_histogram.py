import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_histogram(df, column, plot_frame):
    # Clear old widgets
    for w in plot_frame.winfo_children():
        w.destroy()

    series = df[column].dropna()
    if series.empty:
        from customtkinter import CTkLabel
        CTkLabel(plot_frame, text=f"No valid data in '{column}'").pack(pady=20)
        return

    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)

    ax.hist(series, bins=30, edgecolor='black')
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
