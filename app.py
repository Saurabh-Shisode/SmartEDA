from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import os
import io
import zipfile

from xhtml2pdf import pisa
from flask import make_response, render_template_string


import matplotlib
matplotlib.use('Agg') 

from helpers.file_utils import read_csv_safely
from helpers.preprocessing import preprocess
from helpers.eda_utils import (
    get_eda_code_from_gemini,
    save_all_open_plots,
    exec_and_capture_plots
)

app = Flask(__name__)
app.secret_key = 'secret123'
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/')
def index():
    has_dataset = 'file_path' in session and 'target_var' in session
    return render_template('index.html', has_dataset=has_dataset)


@app.route('/reset')
def reset():
    session.clear()
    return redirect(url_for('index'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        target_var = request.form.get('target_var').strip().lower().replace(" ", "_")

        if file and target_var:
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            session['file_path'] = file_path
            session['target_var'] = target_var

            df = read_csv_safely(file_path)
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

            if df[target_var].dtype in ['int64', 'float64'] and df[target_var].nunique() > 20:
                session['task_type'] = 'regression'
            else:
                session['task_type'] = 'classification'

            return redirect(url_for('preview'))

    return render_template('upload.html')



@app.route('/preview')
def preview():
    file_path = session.get('file_path')
    target_var = session.get('target_var')

    if not file_path or not os.path.exists(file_path):
        return redirect(url_for('upload'))

    df = read_csv_safely(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    preview_data = df.head(10).to_html(classes='data', index=False)
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "target_valid": target_var in df.columns,
        "target_var": target_var,
        "missing_summary": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }

    return render_template('preview.html', preview_data=preview_data, info=info)


@app.route('/eda')
def eda():
    file_path = session.get('file_path')
    target_var = session.get('target_var')
    task_type = session.get('task_type')

    if not file_path or not target_var:
        return redirect(url_for('upload'))

    df = read_csv_safely(file_path)
    df = preprocess(df)

    # Save clean version for modeling
    df.to_csv('static/preprocessed_data.csv', index=False)

    # Make available for Gemini/exec
    globals()['df'] = df
    globals()['data'] = df

    try:
        eda_code = get_eda_code_from_gemini(df, target_var, task_type=task_type)
        clean_code = eda_code.replace("```python", "").replace("```", "").strip()
        exec_and_capture_plots(clean_code)
    except Exception as e:
        print("❌ Error running Gemini code:", e)

    plot_paths = save_all_open_plots(output_dir="static/plots", prefix="plot")
    plot_paths = [path.replace("\\", "/") for path in plot_paths]

    return render_template("eda.html", plot_paths=plot_paths)



@app.route('/eda_report')
def eda_report():
    try:
        from ydata_profiling import ProfileReport
        df = read_csv_safely('static/preprocessed_data.csv')
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        profile = ProfileReport(df, title="SmartEDA Report", minimal=True)
        report_path = os.path.join('static', 'eda_report.html')
        profile.to_file(report_path)

        return redirect(url_for('static', filename='eda_report.html'))
    except Exception as e:
        print("❌ Error generating EDA report:", e)
        return "An error occurred while generating the EDA summary report."


@app.route('/download_all_plots')
def download_all_plots():
    plot_dir = os.path.join('static', 'plots')
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for filename in os.listdir(plot_dir):
            file_path = os.path.join(plot_dir, filename)
            zipf.write(file_path, arcname=filename)

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='eda_plots.zip'
    )


@app.route('/model', methods=['GET', 'POST'])
def model():
    from helpers.model_utils import (
        preprocess_for_modeling,
        run_all_classification_models,
        run_all_regression_models
    )

    try:
        if request.method == 'POST':
            selected_models = request.form.getlist('models')
            session['selected_models'] = selected_models

        if 'model_results' in session:
            return render_template('model_results.html', results=session['model_results'])

        file_path = session.get('file_path')
        target_var = session.get('target_var')
        task_type = session.get('task_type', 'classification')
        selected_models = session.get('selected_models', ['Random Forest', 'Logistic Regression', 'SVM'])

        if not file_path or not target_var:
            return redirect(url_for('upload'))

        df = read_csv_safely('static/preprocessed_data.csv')
        X, y = preprocess_for_modeling(df, target_var)

        if task_type == 'regression':
            model_results = run_all_regression_models(X, y, models_to_run=selected_models)
        else:
            model_results = run_all_classification_models(X, y, models_to_run=selected_models)

        # Identify best model by r2 or f1 score
        best_model_name = None
        best_score = -1
        for name, result in model_results.items():
            score = result['test_metrics']['r2'] if task_type == 'regression' else result['test_metrics']['f1']
            if score > best_score:
                best_score = score
                best_model_name = name

        if task_type == 'classification' and best_model_name:
            session['confusion_matrix'] = {
                'matrix': model_results[best_model_name]['test_metrics']['confusion_matrix'],
                'labels': model_results[best_model_name]['test_metrics']['labels'],
                'model_name': best_model_name
            }

        session['model_results'] = model_results
        return render_template('model_results.html', results=model_results)

    except Exception as e:
        print("❌ Error during model training:", e)
        return "An error occurred while training the models."



@app.route('/download_model_bundle')
def download_model_bundle():
    model_path = 'static/model.pkl'
    scaler_path = 'static/scaler.pkl'

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return "Required files not found.", 404

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        zipf.write(model_path, arcname='model.pkl')
        zipf.write(scaler_path, arcname='scaler.pkl')

    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='SmartEDA_Model_Bundle.zip'
    )


@app.route('/rebuild_model')
def rebuild_model():
    session.pop('model_results', None)
    return redirect(url_for('model'))


@app.route('/download_pdf')
def download_pdf():
    from xhtml2pdf import pisa
    import datetime

    try:
        # Load required data from session
        summary = {}
        file_path = session.get('file_path')
        if file_path and os.path.exists(file_path):
            df = read_csv_safely('static/preprocessed_data.csv')
            df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
            summary = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "missing_summary": df.isnull().sum().to_dict(),
                "dtypes": df.dtypes.astype(str).to_dict()
            }

        results = session.get('model_results', {})
        confusion_matrix = session.get('confusion_matrix', {})
        task_type = session.get('task_type', 'classification')

        # Collect plot paths
        plot_dir = os.path.join('static', 'plots')
        plot_paths = []
        if os.path.exists(plot_dir):
            for f in os.listdir(plot_dir):
                if f.endswith('.png'):
                    plot_paths.append(os.path.join('static', 'plots', f).replace("\\", "/"))

        rendered = render_template(
            'pdf_report.html',
            summary=summary,
            results=results,
            confusion_matrix=confusion_matrix,
            plot_paths=plot_paths,
            task_type=task_type
        )

        # Convert to PDF
        pdf_buffer = io.BytesIO()
        pisa.CreatePDF(io.StringIO(rendered), dest=pdf_buffer)
        pdf_buffer.seek(0)

        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='SmartEDA_Report.pdf'
        )

    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        return "An error occurred while generating the PDF report."



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
