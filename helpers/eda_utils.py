import os
import matplotlib.pyplot as plt
import uuid
import google.generativeai as genai

# ✅ Configure Gemini API
genai.configure(api_key="AIzaSyBjSOhcRqM2IdsM2QbZKBIdVWeX2soOPBc")


def save_all_open_plots(output_dir="static/plots", prefix="plot"):
    os.makedirs(output_dir, exist_ok=True)

    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))

    figs = [plt.figure(i) for i in plt.get_fignums()]
    paths = []

    for fig in figs:
        filename = f"{prefix}_{uuid.uuid4().hex}.png"
        path = os.path.join(output_dir, filename)
        fig.savefig(path)
        paths.append(path.replace("\\", "/"))

    plt.close('all')
    return paths


def build_eda_prompt(data, target_var):
    num_classes = data[target_var].nunique()
    task_type = "classification" if data[target_var].dtype in ['object', 'category', 'int64'] and num_classes <= 20 else "regression"

    column_info = "\n".join([f"- {col}: {dtype}" for col, dtype in data.dtypes.astype(str).items()])
    sample_data = data.head(5).to_dict(orient="records")
    corr = data.corr(numeric_only=True).to_dict()

    prompt = f"""
You are a data analyst assistant.

Dataset information:

Columns and data types:
{column_info}

Sample data (first 5 rows):
{sample_data}

Correlation matrix (numerical columns only):
{corr}

Target column: {target_var}
Task type: {task_type}

Suggest 5-6 useful visualizations using matplotlib or seaborn.

Include:
- Numeric distributions
- Categorical value counts
- {"Target-class balance" if task_type == "classification" else "Target value distribution"}
- Feature-target relationships

Do not include correlation heatmaps or correlation matrix analysis yet,
as categorical features have not been numerically encoded.

Return clean matplotlib or seaborn Python code only.

Important:
- Use plt.figure() for each plot.
- Do NOT call plt.show()
- Do NOT call plt.savefig()
- Leave the figures open after plotting.
"""
    return prompt



def exec_and_capture_plots(code_str):
    local_scope = {}
    exec(code_str, globals(), local_scope)

    if not plt.get_fignums():
        plt.figure()
        plt.plot([0, 1, 2], [3, 4, 5])
        plt.title("Dummy plot")

    return plt.get_fignums()


def get_eda_code_from_gemini(df, target_var, task_type='classification'):
    prompt = build_eda_prompt(df, target_var)
    model = genai.GenerativeModel("models/gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text

