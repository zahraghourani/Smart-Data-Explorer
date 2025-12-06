from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from customtkinter import CTkLabel

def plot_scatter(df, x_col, y_col, plot_frame):
    for w in plot_frame.winfo_children():
        w.destroy()

    if x_col == y_col:
        CTkLabel(plot_frame, text="Select two different numeric columns.").pack(pady=20)
        return

    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)

    ax.scatter(df[x_col], df[y_col])
    ax.set_title(f"Scatter: {x_col} vs {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
