import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def maxout(x, w1=1.0, b1=0.0, w2=0.5, b2=-1.0):
    """Simplified maxout with 2 linear units"""
    return np.maximum(w1 * x + b1, w2 * x + b2)

def maxout_derivative(x, w1=1.0, b1=0.0, w2=0.5, b2=-1.0):
    linear1 = w1 * x + b1
    linear2 = w2 * x + b2
    return np.where(linear1 > linear2, w1, w2)

def softmax(x):
    """Softmax for visualization - applies along the array"""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

def softmax_derivative(x):
    """Jacobian diagonal elements for visualization"""
    s = softmax(x)
    return s * (1 - s)

def get_function_info(func_name, alpha=1.0, w1=1.0, b1=0.0, w2=0.5, b2=-1.0):
    if func_name == "Sigmoid":
        formula_html = r"""
        <div style="font-size: 16px; padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 10px 0;">
            <strong>Formula:</strong><br>
            <span style="font-size: 18px;">σ(x) = <sup>1</sup>⁄<sub>(1 + e<sup>−x</sup>)</sub></span>
        </div>
        """
        advantages = """• Smooth and differentiable everywhere
• Output bounded in (0,1) range - interpretable as probabilities
• Historically well-studied"""
        disadvantages = """• Vanishing gradient problem for very high/low values
• Not zero-centered (outputs always positive)
• Computationally expensive (exponential calculation)"""
        use_cases = """• Output layer for binary classification tasks
• Gate mechanisms in LSTM and GRU cells
• Attention mechanisms in transformers
• Legacy neural networks (pre-2010s)
• Any scenario requiring probability interpretation"""
    elif func_name == "Tanh":
        formula_html = r"""
        <div style="font-size: 16px; padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 10px 0;">
            <strong>Formula:</strong><br>
            <span style="font-size: 18px;">tanh(x) = <sup>(e<sup>2x</sup> − 1)</sup>⁄<sub>(e<sup>2x</sup> + 1)</sub></span>
        </div>
        """
        advantages = """• Zero-centered output in (-1,1) range
• Smooth gradient
• Stronger gradients than sigmoid (derivatives steeper)"""
        disadvantages = """• Still suffers from vanishing gradient problem
• Computationally expensive (exponential calculation)
• Can saturate for large input values"""
        use_cases = """• Hidden layers in Recurrent Neural Networks (RNNs)
• LSTM and GRU cell activations
• When zero-centered outputs are beneficial
• Preferable to sigmoid in hidden layers
• Time series analysis and sequence modeling"""
    elif func_name == "ReLU":
        formula_html = r"""
        <div style="font-size: 16px; padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 10px 0;">
            <strong>Formula:</strong><br>
            <span style="font-size: 18px;">ReLU(x) = max(0, x)</span>
        </div>
        """
        advantages = """• Extremely computationally efficient
• Significantly reduces vanishing gradient
• Induces sparsity (many neurons output zero)
• Fast convergence in training"""
        disadvantages = """• Dying ReLU problem (neurons can become permanently inactive)
• Not zero-centered
• Unbounded output
• Non-differentiable at x=0"""
        use_cases = """• Default choice for hidden layers in feedforward networks
• Convolutional Neural Networks (CNNs) for computer vision
• Most modern deep learning architectures
• ResNets, VGGNet, and other vision models
• Works well in very deep networks"""
    elif func_name == "Leaky ReLU":
        formula_html = rf"""
        <div style="font-size: 16px; padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 10px 0;">
            <strong>Formula:</strong><br>
            <span style="font-size: 18px;">LeakyReLU(x) = max({alpha}x, x)</span>
        </div>
        """
        advantages = """• Addresses the dying ReLU problem
• Allows small negative gradient (no dead neurons)
• Computationally efficient
• Better gradient flow than ReLU"""
        disadvantages = """• Can produce inconsistent predictions for negative inputs
• Requires tuning alpha hyperparameter
• Not as widely adopted as ReLU"""
        use_cases = """• Deep networks where dying ReLU is problematic
• Generative Adversarial Networks (GANs)
• When negative values should have some influence
• Alternative to ReLU when training instability occurs
• Networks with many layers"""
    elif func_name == "ELU":
        formula_html = rf"""
        <div style="font-size: 16px; padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 10px 0;">
            <strong>Formula:</strong><br>
            <span style="font-size: 18px;">
                ELU(x) = {{ x &nbsp;&nbsp; if x ≥ 0<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                {alpha}(e<sup>x</sup> − 1) &nbsp;&nbsp; if x < 0
            </span>
        </div>
        """
        advantages = """• Smooth everywhere (helps optimization)
• Negative saturation reduces bias shift
• Can produce negative outputs (self-normalizing)
• Pushes mean activations closer to zero"""
        disadvantages = """• Computationally expensive due to exponential
• Requires tuning alpha parameter
• Slower than ReLU in practice"""
        use_cases = """• Self-normalizing neural networks (SNNs)
• When smooth, continuous activation is needed
• Deep networks requiring stable training
• Alternative to ReLU with better properties
• Networks sensitive to bias shift"""
    elif func_name == "Maxout":
        formula_html = rf"""
        <div style="font-size: 16px; padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 10px 0;">
            <strong>Formula:</strong><br>
            <span style="font-size: 18px;">
                Maxout(x) = max(w₁<sup>T</sup>x + b₁, w₂<sup>T</sup>x + b₂)<br>
                where w₁ = {w1}, b₁ = {b1}, w₂ = {w2}, b₂ = {b2}
            </span>
        </div>
        """
        advantages = """• Can learn the activation function
• Generalizes ReLU and Leaky ReLU
• No dying neuron problem
• Flexible and powerful"""
        disadvantages = """• Doubles the number of parameters per neuron
• More complex to implement
• Higher memory and computational requirements
• Less commonly used in practice"""
        use_cases = """• Research applications exploring learned activations
• Networks with dropout (shown to work well together)
• When computational resources allow for flexibility
• Specialized architectures requiring adaptive activations
• Scenarios where standard activations underperform"""
    elif func_name == "Softmax":
        formula_html = r"""
        <div style="font-size: 16px; padding: 10px; background: #f8f9fa; border-radius: 5px; margin: 10px 0;">
            <strong>Formula:</strong><br>
            <span style="font-size: 18px;">
                Softmax(x<sub>i</sub>) = <sup>e<sup>x<sub>i</sub></sup></sup>⁄<sub>Σ<sub>j</sub> e<sup>x<sub>j</sub></sup></sub>
            </span>
        </div>
        """
        advantages = """• Outputs sum to 1 - true probability distribution
• Differentiable everywhere
• Highlights maximum value while suppressing others
• Theoretically well-grounded (information theory)"""
        disadvantages = """• Computationally expensive (multiple exponentials)
• Can produce overconfident predictions
• Sensitive to outliers in input
• Requires normalization across all outputs simultaneously"""
        use_cases = """• Multi-class classification output layer
• Attention mechanisms in transformers
• Neural machine translation
• Any task requiring probability distribution over classes
• Reinforcement learning policy networks"""
    
    return formula_html, advantages, disadvantages, use_cases

def create_visualization(func_name, x_range=(-10, 10), alpha=1.0, w1=1.0, b1=0.0, w2=0.5, b2=-1.0):
    if func_name == "Softmax":
        # Create interactive softmax visualization
        return create_softmax_network_viz()
    
    x = np.linspace(x_range[0], x_range[1], 1000)
    
    # Select function based on name
    if func_name == "Sigmoid":
        y = sigmoid(x)
        y_prime = sigmoid_derivative(x)
    elif func_name == "Tanh":
        y = tanh(x)
        y_prime = tanh_derivative(x)
    elif func_name == "ReLU":
        y = relu(x)
        y_prime = relu_derivative(x)
    elif func_name == "Leaky ReLU":
        y = leaky_relu(x, alpha)
        y_prime = leaky_relu_derivative(x, alpha)
    elif func_name == "ELU":
        y = elu(x, alpha)
        y_prime = elu_derivative(x, alpha)
    elif func_name == "Maxout":
        y = maxout(x, w1, b1, w2, b2)
        y_prime = maxout_derivative(x, w1, b1, w2, b2)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Activation Function', 'First Derivative'),
        horizontal_spacing=0.12
    )
    
    # Function plot
    fig.add_trace(
        go.Scatter(x=x, y=y, name=func_name, line=dict(color='#1f77b4', width=3)),
        row=1, col=1
    )
    
    # First derivative
    fig.add_trace(
        go.Scatter(x=x, y=y_prime, name="f'(x)", line=dict(color='#ff7f0e', width=3)),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="x", row=1, col=1, gridcolor='lightgray')
    fig.update_xaxes(title_text="x", row=1, col=2, gridcolor='lightgray')
    
    fig.update_yaxes(title_text="f(x)", row=1, col=1, gridcolor='lightgray')
    fig.update_yaxes(title_text="f'(x)", row=1, col=2, gridcolor='lightgray')
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text=f"{func_name} Activation Function",
        title_x=0.5,
        title_font_size=20,
        plot_bgcolor='white'
    )
    
    return fig

def create_softmax_network_viz():
    """Create an interactive network visualization for Softmax"""
    fig = go.Figure()
    
    # Define positions
    input_nodes = 4
    hidden_nodes = 4
    output_nodes = 4
    
    # Input layer positions
    input_y = np.linspace(0, 10, input_nodes)
    input_x = np.ones(input_nodes) * 1
    
    # Hidden layer (z) positions
    hidden_y = np.linspace(0, 10, hidden_nodes)
    hidden_x = np.ones(hidden_nodes) * 4
    
    # Softmax layer positions
    softmax_x = np.ones(hidden_nodes) * 7
    softmax_y = hidden_y
    
    # Output probabilities positions
    output_y = np.linspace(1, 9, output_nodes)
    output_x = np.ones(output_nodes) * 10
    
    # Sample values
    z_values = np.array([2.5, 1.8, 0.5, 3.2])
    exp_z = np.exp(z_values - np.max(z_values))
    probabilities = exp_z / np.sum(exp_z)
    
    colors = ['#7cb342', '#1976d2', '#7b1fa2', '#d32f2f']
    labels = ['green', 'blue', 'purple', 'red']
    
    # Draw connections from input to hidden
    for i in range(input_nodes):
        for j in range(hidden_nodes):
            opacity = 0.15 if i != j else 0.4
            color = colors[j] if i == j else 'gray'
            fig.add_trace(go.Scatter(
                x=[input_x[i], hidden_x[j]],
                y=[input_y[i], hidden_y[j]],
                mode='lines',
                line=dict(color=color, width=2, dash='solid'),
                opacity=opacity,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Draw connections from hidden to softmax
    for i in range(hidden_nodes):
        fig.add_trace(go.Scatter(
            x=[hidden_x[i], softmax_x[i]],
            y=[hidden_y[i], softmax_y[i]],
            mode='lines',
            line=dict(color=colors[i], width=3),
            opacity=0.6,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Draw connections from softmax to output
    for i in range(output_nodes):
        fig.add_trace(go.Scatter(
            x=[softmax_x[i], output_x[i]],
            y=[softmax_y[i], output_y[i]],
            mode='lines',
            line=dict(color=colors[i], width=2),
            opacity=0.5,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add bias node
    fig.add_trace(go.Scatter(
        x=[1],
        y=[11.5],
        mode='markers+text',
        marker=dict(size=35, color='white', line=dict(color='black', width=2)),
        text=['bias'],
        textposition='middle center',
        textfont=dict(size=10),
        showlegend=False,
        hoverinfo='text',
        hovertext='Bias term added to weighted sum'
    ))
    
    # Bias connections
    for j in range(hidden_nodes):
        fig.add_trace(go.Scatter(
            x=[1, hidden_x[j]],
            y=[11.5, hidden_y[j]],
            mode='lines',
            line=dict(color=colors[j], width=2),
            opacity=0.3,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Input nodes
    for i in range(input_nodes):
        label = f'x<sub>{i+1}</sub>' if i < 3 else 'x<sub>n</sub>'
        fig.add_trace(go.Scatter(
            x=[input_x[i]],
            y=[input_y[i]],
            mode='markers+text',
            marker=dict(size=40, color='white', line=dict(color='black', width=2)),
            text=[label],
            textposition='middle center',
            textfont=dict(size=12),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'Input feature {i+1}'
        ))
    
    # Hidden nodes (z)
    for i in range(hidden_nodes):
        label = f'z<sub>{i+1}</sub>' if i < 3 else f'z<sub>K</sub>'
        hover_text = f'z<sub>{i+1}</sub> = w<sub>{i+1}</sub><sup>T</sup>x = {z_values[i]:.2f}'
        fig.add_trace(go.Scatter(
            x=[hidden_x[i]],
            y=[hidden_y[i]],
            mode='markers+text',
            marker=dict(size=40, color='lightblue', line=dict(color='black', width=2)),
            text=[label],
            textposition='middle center',
            textfont=dict(size=12),
            showlegend=False,
            hoverinfo='text',
            hovertext=hover_text
        ))
    
    # Softmax computation nodes
    for i in range(hidden_nodes):
        exp_val = exp_z[i]
        prob_val = probabilities[i]
        numerator = f'e<sup>{z_values[i]:.1f}</sup>'
        hover_text = f'Softmax: {numerator} / Σe<sup>z</sup> = {prob_val:.3f}'
        
        fig.add_trace(go.Scatter(
            x=[softmax_x[i]],
            y=[softmax_y[i]],
            mode='markers+text',
            marker=dict(size=50, color='#c8e6c9', line=dict(color='black', width=2)),
            text=[f'{prob_val:.2f}'],
            textposition='middle center',
            textfont=dict(size=10),
            showlegend=False,
            hoverinfo='text',
            hovertext=hover_text
        ))
    
    # Output probability bars
    max_prob = np.max(probabilities)
    for i in range(output_nodes):
        bar_length = probabilities[i] * 2
        fig.add_trace(go.Scatter(
            x=[output_x[i], output_x[i] + bar_length],
            y=[output_y[i], output_y[i]],
            mode='lines+markers+text',
            line=dict(color=colors[i], width=20),
            marker=dict(size=8, color=colors[i]),
            text=['', f'{probabilities[i]:.3f}'],
            textposition='middle right',
            textfont=dict(size=11, color=colors[i]),
            showlegend=False,
            hoverinfo='text',
            hovertext=f'{labels[i]}: {probabilities[i]:.3f} ({probabilities[i]*100:.1f}%)'
        ))
        
        # Add label
        fig.add_trace(go.Scatter(
            x=[output_x[i] + bar_length/2],
            y=[output_y[i]],
            mode='text',
            text=[labels[i]],
            textposition='middle center',
            textfont=dict(size=12, color='white', family='Arial Black'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add annotations
    fig.add_annotation(x=1, y=-1, text="<b>Inputs</b>", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=4, y=-1, text="<b>z<sub>j</sub> = w<sub>j</sub><sup>T</sup>x</b>", showarrow=False, font=dict(size=14))
    fig.add_annotation(x=7, y=-1, text="<b>SoftMax</b>", showarrow=False, font=dict(size=14), 
                      bgcolor='#c8e6c9', bordercolor='black', borderwidth=2, borderpad=4)
    fig.add_annotation(x=11, y=-1, text="<b>Probabilities</b>", showarrow=False, font=dict(size=14))
    
    # Add softmax formula annotation
    fig.add_annotation(
        x=7, y=12,
        text="Σ probabilities = 1.000",
        showarrow=False,
        font=dict(size=12),
        bgcolor='lightyellow',
        bordercolor='black',
        borderwidth=1
    )
    
    fig.update_layout(
        title=dict(
            text="Softmax Activation: Multi-Class Classification Network",
            x=0.5,
            font=dict(size=20)
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 13]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 13]),
        plot_bgcolor='white',
        height=600,
        hovermode='closest'
    )
    
    return fig

def create_info_html(advantages, disadvantages, use_cases):
    html = f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0;">
        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
            <h3 style="margin-top: 0; color: #2e7d32;">✓ Advantages</h3>
            <div style="line-height: 1.8;">{advantages.replace(chr(10), '<br>')}</div>
        </div>
        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
            <h3 style="margin-top: 0; color: #c62828;">✗ Disadvantages</h3>
            <div style="line-height: 1.8;">{disadvantages.replace(chr(10), '<br>')}</div>
        </div>
        <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
            <h3 style="margin-top: 0; color: #1565c0;">⚡ Use Cases</h3>
            <div style="line-height: 1.8;">{use_cases.replace(chr(10), '<br>')}</div>
        </div>
    </div>
    """
    return html

# Create Gradio interface
with gr.Blocks(title="Neural Network Activation Functions Visualizer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🧠 Neural Network Activation Functions Visualizer
    
    Explore different activation functions with their derivatives, advantages, disadvantages, and use cases.
    """)
    
    func_dropdown = gr.Dropdown(
        choices=["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "ELU", "Maxout", "Softmax"],
        value="Sigmoid",
        label="Activation Function"
    )
    
    formula_display = gr.HTML()
    info_display = gr.HTML()
    
    plot = gr.Plot(show_label=False)
    
    gr.Markdown("### Adjust Parameters")
    
    with gr.Row():
        with gr.Column():
            x_min = gr.Slider(-20, 0, value=-10, label="X Min", step=1)
            x_max = gr.Slider(0, 20, value=10, label="X Max", step=1)
        
        with gr.Column():
            alpha = gr.Slider(0.01, 2.0, value=1.0, label="Alpha (for Leaky ReLU & ELU)", step=0.01)
    
    with gr.Row():
        with gr.Column():
            w1 = gr.Slider(-2, 2, value=1.0, label="w₁ (Maxout)", step=0.1)
            b1 = gr.Slider(-2, 2, value=0.0, label="b₁ (Maxout)", step=0.1)
        
        with gr.Column():
            w2 = gr.Slider(-2, 2, value=0.5, label="w₂ (Maxout)", step=0.1)
            b2 = gr.Slider(-2, 2, value=-1.0, label="b₂ (Maxout)", step=0.1)
    
    # Update plots when any input changes
    inputs = [func_dropdown, x_min, x_max, alpha, w1, b1, w2, b2]
    
    def update_all(func, x_min, x_max, alpha, w1, b1, w2, b2):
        viz = create_visualization(func, (x_min, x_max), alpha, w1, b1, w2, b2)
        formula_html, advantages, disadvantages, use_cases = get_function_info(func, alpha, w1, b1, w2, b2)
        info_html = create_info_html(advantages, disadvantages, use_cases)
        return viz, formula_html, info_html
    
    for input_component in inputs:
        input_component.change(
            fn=update_all,
            inputs=inputs,
            outputs=[plot, formula_display, info_display]
        )
    
    # Initial plot
    demo.load(
        fn=update_all,
        inputs=inputs,
        outputs=[plot, formula_display, info_display]
    )
    
    gr.Markdown("""
    ---
    ### About the Derivative:
    The **First Derivative (f'(x))** shows the gradient/slope of the activation function, which is crucial for backpropagation during neural network training.
    """)

if __name__ == "__main__":
    demo.launch()
