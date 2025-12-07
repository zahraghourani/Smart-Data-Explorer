import customtkinter as ctk
from tkinter import messagebox
from PIL import Image
import base64
import io
import pandas as pd
import numpy as np

from GUI.utils.data_loader import DataStorage
from ML.model_training import (
    get_dataset_info,
    validate_target_column,
    encode_target_if_categorical,
    split_dataset_fixed,
    auto_model_selection,
    evaluate_model_metrics,
    feature_engineering,
)


class ModelPage(ctk.CTkFrame):
    """
    ML Modeling Dashboard page.
    - User chooses target and features.
    - Auto model selection + metrics.
    - Shows plots generated in evaluate_model_metrics().
    - Allows predicting a new sample based on original feature values.
    """

    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.controller = controller

        # global dataset info (set in refresh)
        self.df = None
        self.columns = []
        self.feature_vars = {}

        # keep references to images so Tkinter doesn't GC them
        self.left_plot_img = None
        self.right_plot_img = None

        # model + metadata for prediction
        self.best_model = None
        self.trained_feature_names = []      # columns after feature_engineering
        self.problem_type_current = None
        self.target_encoder = None

        # keep original features and training df for FE reuse
        self.selected_features_current = []  # original feature names selected in UI
        self.df_for_fe = None               # encoded df passed to FE during training
        self.target_name_current = None

        # ---- White background for page ----
        self.configure(fg_color="white")

        # ====== SCROLLABLE CONTENT AREA ======
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="white")
        self.scroll.pack(fill="both", expand=True)

        self.scroll.grid_rowconfigure(0, weight=0)
        self.scroll.grid_rowconfigure(1, weight=1)
        self.scroll.grid_rowconfigure(2, weight=1)
        self.scroll.grid_columnconfigure(0, weight=1)
        self.scroll.grid_columnconfigure(1, weight=1)

        # ───────── HEADER ─────────
        header = ctk.CTkFrame(self.scroll, fg_color="white")
        header.grid(row=0, column=0, columnspan=2,
                    sticky="ew", pady=(10, 5), padx=20)
        header.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            header,
            text="ML Modeling Dashboard",
            font=("Arial", 24, "bold"),
            fg_color="white"
        )
        title_label.grid(row=0, column=0, sticky="w")

        # ───────── MIDDLE ROW: CONFIG (L) + PERFORMANCE (R) ─────────
        # CONFIG CARD
        self.config_card = ctk.CTkFrame(
            self.scroll, corner_radius=20, border_width=1, fg_color="#F4F4F4"
        )
        self.config_card.grid(row=1, column=0, sticky="nsew",
                              padx=(20, 10), pady=10)
        self.config_card.grid_rowconfigure(8, weight=1)
        self.config_card.grid_columnconfigure(0, weight=1)

        config_title = ctk.CTkLabel(
            self.config_card,
            text="Model Configuration",
            font=("Arial", 18, "bold"),
            fg_color="#F4F4F4"
        )
        config_title.grid(row=0, column=0, sticky="w",
                          padx=20, pady=(15, 10))

        # Target selector
        target_label = ctk.CTkLabel(
            self.config_card,
            text="Target Column Selector",
            font=("Arial", 14, "bold"),
            fg_color="#F4F4F4"
        )
        target_label.grid(row=1, column=0, sticky="w", padx=20)

        self.target_var = ctk.StringVar(value="")
        self.target_dropdown = ctk.CTkOptionMenu(
            self.config_card,
            values=["(load CSV first)"],
            variable=self.target_var,
            width=280,
            state="disabled",
            command=self.on_target_changed
        )
        self.target_dropdown.grid(row=2, column=0, padx=20,
                                  pady=(5, 10), sticky="w")

        # Feature selector (multi-select)
        features_label = ctk.CTkLabel(
            self.config_card,
            text="Feature Columns (inputs):",
            font=("Arial", 13, "bold"),
            fg_color="#F4F4F4"
        )
        features_label.grid(row=3, column=0, sticky="w",
                            padx=20, pady=(5, 2))

        self.features_frame = ctk.CTkScrollableFrame(
            self.config_card, height=140, fg_color="#E7E7E7"
        )
        self.features_frame.grid(row=4, column=0, sticky="nsew",
                                 padx=20, pady=(0, 10))

        # Problem type display
        self.problem_type_var = ctk.StringVar(value="Problem Type: —")
        problem_type_label = ctk.CTkLabel(
            self.config_card,
            textvariable=self.problem_type_var,
            font=("Arial", 13),
            fg_color="#F4F4F4"
        )
        problem_type_label.grid(row=5, column=0, padx=20,
                                pady=(5, 10), sticky="w")

        # Train button
        self.train_button = ctk.CTkButton(
            self.config_card,
            text="Train Model",
            height=38,
            command=self.on_train_clicked
        )
        self.train_button.grid(row=6, column=0, padx=20,
                               pady=(5, 5), sticky="w")

        # Predict button (enabled after training)
        self.predict_button = ctk.CTkButton(
            self.config_card,
            text="Predict New Sample",
            height=38,
            state="disabled",
            command=self.on_predict_clicked
        )
        self.predict_button.grid(row=7, column=0, padx=20,
                                 pady=(0, 10), sticky="w")

        # Status text
        self.status_label = ctk.CTkLabel(
            self.config_card,
            text="Please load a CSV from the welcome page.",
            wraplength=360,
            justify="left",
            fg_color="#F4F4F4"
        )
        self.status_label.grid(row=8, column=0, padx=20,
                               pady=(5, 15), sticky="nw")

        # PERFORMANCE CARD (right)
        perf_card = ctk.CTkFrame(
            self.scroll, corner_radius=20, border_width=1, fg_color="#F4F4F4"
        )
        perf_card.grid(row=1, column=1, sticky="nsew",
                       padx=(10, 20), pady=10)
        perf_card.grid_rowconfigure(2, weight=1)
        perf_card.grid_columnconfigure(0, weight=1)

        perf_title = ctk.CTkLabel(
            perf_card,
            text="Model Performance",
            font=("Arial", 18, "bold"),
            fg_color="#F4F4F4"
        )
        perf_title.grid(row=0, column=0, sticky="w",
                        padx=20, pady=(15, 5))

        # Summary box
        summary_frame = ctk.CTkFrame(perf_card, corner_radius=16,
                                     fg_color="#E7E7E7")
        summary_frame.grid(row=1, column=0, sticky="ew",
                           padx=20, pady=(5, 10))
        summary_frame.grid_columnconfigure(0, weight=1)

        self.summary_text = ctk.StringVar(
            value="No model trained yet.\nSelect a target and click Train."
        )
        summary_label = ctk.CTkLabel(
            summary_frame,
            textvariable=self.summary_text,
            justify="left",
            anchor="w",
            fg_color="#E7E7E7"
        )
        summary_label.grid(row=0, column=0, padx=15,
                           pady=10, sticky="w")

        # Detailed metrics box (read-only)
        metrics_frame = ctk.CTkFrame(perf_card, corner_radius=16,
                                     fg_color="#E7E7E7")
        metrics_frame.grid(row=2, column=0, sticky="nsew",
                           padx=20, pady=(5, 15))
        metrics_frame.grid_rowconfigure(1, weight=1)
        metrics_frame.grid_columnconfigure(0, weight=1)

        metrics_label = ctk.CTkLabel(
            metrics_frame,
            text="Evaluation Metrics",
            font=("Arial", 14, "bold"),
            fg_color="#E7E7E7"
        )
        metrics_label.grid(row=0, column=0, sticky="w",
                           padx=15, pady=(10, 0))

        self.metrics_box = ctk.CTkTextbox(metrics_frame)
        self.metrics_box.grid(row=1, column=0, sticky="nsew",
                              padx=15, pady=(5, 15))
        self.metrics_box.configure(state="disabled")  # read-only

        # ───────── BOTTOM ROW: PLOTS ─────────
        plot_card = ctk.CTkFrame(self.scroll, corner_radius=20, border_width=1,
                                 fg_color="#F4F4F4")
        plot_card.grid(row=2, column=0, sticky="nsew",
                       padx=(20, 10), pady=(0, 20))
        plot_card.grid_rowconfigure(1, weight=1)
        plot_card.grid_columnconfigure(0, weight=1)

        plot_title = ctk.CTkLabel(
            plot_card,
            text="Predicted vs Actual / Confusion Matrix",
            font=("Arial", 14, "bold"),
            fg_color="#F4F4F4"
        )
        plot_title.grid(row=0, column=0, sticky="w",
                        padx=20, pady=(15, 5))

        self.plot_label_left = ctk.CTkLabel(
            plot_card,
            text="No plot yet.\nTrain a model first.",
            justify="center",
            fg_color="#F4F4F4"
        )
        self.plot_label_left.grid(row=1, column=0, padx=20,
                                  pady=(0, 15), sticky="nsew")

        feat_card = ctk.CTkFrame(self.scroll, corner_radius=20, border_width=1,
                                 fg_color="#F4F4F4")
        feat_card.grid(row=2, column=1, sticky="nsew",
                       padx=(10, 20), pady=(0, 20))
        feat_card.grid_rowconfigure(1, weight=1)
        feat_card.grid_columnconfigure(0, weight=1)

        feat_title = ctk.CTkLabel(
            feat_card,
            text="Feature Importance / Coefficients / ROC",
            font=("Arial", 14, "bold"),
            fg_color="#F4F4F4"
        )
        feat_title.grid(row=0, column=0, sticky="w",
                        padx=20, pady=(15, 5))

        self.feat_label_right = ctk.CTkLabel(
            feat_card,
            text="No plot yet.",
            justify="center",
            fg_color="#F4F4F4"
        )
        self.feat_label_right.grid(row=1, column=0, padx=20,
                                   pady=(0, 15), sticky="nsew")

    # ───────── helpers ─────────
    def build_feature_checkboxes(self, target_name: str):
        """Rebuild list of feature checkboxes when dataset/target changes."""
        for w in self.features_frame.winfo_children():
            w.destroy()
        self.feature_vars.clear()

        for col in self.columns:
            if col == target_name:
                continue
            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                self.features_frame,
                text=col,
                variable=var
            )
            cb.pack(anchor="w", padx=5, pady=2)
            self.feature_vars[col] = var

    def on_target_changed(self, new_target: str):
        if self.columns:
            self.build_feature_checkboxes(new_target)

    def _b64_to_ctkimage(self, b64_string, max_size=(420, 260)):
        """Convert base64 PNG string from evaluate_model_metrics to CTkImage."""
        if not b64_string:
            return None
        try:
            img_bytes = base64.b64decode(b64_string)
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img.thumbnail(max_size, Image.LANCZOS)
            return ctk.CTkImage(light_image=pil_img,
                                dark_image=pil_img,
                                size=pil_img.size)
        except Exception as e:
            print("Error decoding image:", e)
            return None

    def _update_plots(self, problem_type, metrics):
        """Show backend plots in bottom cards."""

        # ===== CLASSIFICATION =====
        if problem_type == "classification":
            # ---- Left: confusion matrix ----
            cm_b64 = metrics.get("confusion_matrix_plot")
            cm_img = self._b64_to_ctkimage(cm_b64)
            self.left_plot_img = cm_img

            if cm_img is not None:
                self.plot_label_left.configure(image=cm_img, text="")
            else:
                self.plot_label_left.configure(
                    image=None,
                    text="Confusion matrix not available."
                )

            # ---- Right: prefer feature importance, else ROC ----
            feat_b64 = metrics.get("feature_importance_plot")
            roc_b64 = metrics.get("roc_curve_plot")

            right_img = None
            if feat_b64 is not None:
                right_img = self._b64_to_ctkimage(feat_b64)
            elif roc_b64 is not None:
                right_img = self._b64_to_ctkimage(roc_b64)

            self.right_plot_img = right_img

            if right_img is not None:
                self.feat_label_right.configure(image=right_img, text="")
            else:
                # Choose appropriate message
                if "feature_importance_plot" in metrics:
                    msg = "Feature importance could not be rendered."
                elif "roc_curve_plot" in metrics:
                    msg = "ROC curve could not be rendered."
                else:
                    msg = (
                        "No feature importance or ROC curve available.\n"
                        "ROC requires binary classification with predict_proba."
                    )
                self.feat_label_right.configure(image=None, text=msg)

        # ===== REGRESSION =====
        else:  # regression
            # ---- Left: Predicted vs Actual ----
            pred_b64 = metrics.get("predicted_vs_actual_plot")
            pred_img = self._b64_to_ctkimage(pred_b64)
            self.left_plot_img = pred_img

            if pred_img is not None:
                self.plot_label_left.configure(image=pred_img, text="")
            else:
                self.plot_label_left.configure(
                    image=None,
                    text="Predicted vs Actual plot not available."
                )

            # ---- Right: Feature importance (if available) ----
            feat_b64 = metrics.get("feature_importance_plot")
            feat_img = self._b64_to_ctkimage(feat_b64)
            self.right_plot_img = feat_img

            if feat_img is not None:
                self.feat_label_right.configure(image=feat_img, text="")
            else:
                self.feat_label_right.configure(
                    image=None,
                    text="Feature importance not available\nfor this model."
                )

    # ───────── lifecycle ─────────
    def refresh(self):
        """Called when we switch to this page."""
        if not DataStorage.loaded or DataStorage.df is None:
            self.df = None
            self.columns = []
            self.target_dropdown.configure(values=["(load CSV first)"],
                                           state="disabled")
            self.target_var.set("(load CSV first)")
            self.problem_type_var.set("Problem Type: —")
            self.status_label.configure(
                text="Please go to the Welcome page and load a CSV file."
            )
            self.build_feature_checkboxes("")
            self.predict_button.configure(state="disabled")
            return

        self.df = DataStorage.df
        info = get_dataset_info(self.df)
        self.columns = info["columns"]

        self.target_dropdown.configure(values=self.columns, state="normal")
        if self.columns:
            self.target_var.set(self.columns[0])
            self.build_feature_checkboxes(self.columns[0])

        self.status_label.configure(
            text="Choose a target column, select features, and click Train."
        )
        self.problem_type_var.set("Problem Type: will be detected after training.")

    # ───────── train button ─────────
    def on_train_clicked(self):
        if self.df is None:
            messagebox.showerror(
                "Error", "No dataset loaded. Please load a CSV first.")
            return

        target = self.target_var.get()
        if not target or target not in self.columns:
            messagebox.showerror(
                "Error", "Please select a valid target column.")
            return

        # which features?
        selected_features = [c for c, v in self.feature_vars.items() if v.get()]
        if not selected_features:
            messagebox.showerror(
                "Error", "Please select at least one feature column.")
            return

        if target in selected_features:
            selected_features.remove(target)

        is_valid, msg = validate_target_column(self.df, target)
        if not is_valid:
            messagebox.showerror("Invalid target", msg)
            self.status_label.configure(text=msg)
            return

        self.status_label.configure(
            text=f"Target '{target}' is valid. Training models...")

        # subset to selected features + target
        df_sel = self.df[selected_features + [target]].copy()

        # encode target if needed
        df_encoded, encoder, problem_type = encode_target_if_categorical(
            df_sel, target
        )

        # split + FE inside
        X_train, X_test, y_train, y_test = split_dataset_fixed(
            df_encoded, target
        )

        best_model, performance = auto_model_selection(
            X_train, X_test, y_train, y_test, problem_type
        )
        metrics = evaluate_model_metrics(
            best_model, X_test, y_test, problem_type
        )

        # Save model + metadata for prediction
        self.best_model = best_model
        self.trained_feature_names = X_train.columns.tolist()
        self.problem_type_current = problem_type
        self.target_encoder = encoder

        self.selected_features_current = selected_features
        self.df_for_fe = df_encoded.copy()
        self.target_name_current = target

        self.predict_button.configure(state="normal")  # enable prediction

        # update plots
        self._update_plots(problem_type, metrics)

        # update problem type label
        self.problem_type_var.set(
            f"Problem Type: {problem_type.capitalize()}")

        # summary card
        best_name = max(performance, key=performance.get)
        if problem_type == "classification":
            summ = (
                f"Best model ({best_name}):\n"
                f"Accuracy: {metrics['accuracy']:.3f}\n"
                f"Precision (weighted): {metrics['precision']:.3f}\n"
                f"Recall (weighted): {metrics['recall']:.3f}\n"
                f"F1-score (weighted): {metrics['f1_score']:.3f}"
            )
        else:
            summ = (
                f"Best model ({best_name}):\n"
                f"MAE: {metrics['MAE']:.3f}\n"
                f"RMSE: {metrics['RMSE']:.3f}\n"
                f"R²: {metrics['R2']:.3f}"
            )
        self.summary_text.set(summ)

        # detailed metrics text (read-only)
        self.metrics_box.configure(state="normal")
        self.metrics_box.delete("1.0", "end")
        self.metrics_box.insert("end", f"Problem type: {problem_type}\n")
        self.metrics_box.insert("end", "Model scores (selection phase):\n")
        for name, score in performance.items():
            self.metrics_box.insert("end", f"  - {name}: {score:.4f}\n")

        self.metrics_box.insert("end", "\nBest model metrics:\n")
        if problem_type == "classification":
            self.metrics_box.insert("end", f"Accuracy: {metrics['accuracy']:.4f}\n")
            self.metrics_box.insert("end", f"Precision (weighted): {metrics['precision']:.4f}\n")
            self.metrics_box.insert("end", f"Recall (weighted): {metrics['recall']:.4f}\n")
            self.metrics_box.insert("end", f"F1-score (weighted): {metrics['f1_score']:.4f}\n")
            if "roc_auc" in metrics:
                self.metrics_box.insert("end", f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        else:
            self.metrics_box.insert("end", f"MAE: {metrics['MAE']:.4f}\n")
            self.metrics_box.insert("end", f"MSE: {metrics['MSE']:.4f}\n")
            self.metrics_box.insert("end", f"RMSE: {metrics['RMSE']:.4f}\n")
            self.metrics_box.insert("end", f"R²: {metrics['R2']:.4f}\n")
        self.metrics_box.configure(state="disabled")

        self.status_label.configure(
            text=f"Training complete. Best model trained on '{target}'.")

    # ───────── Predict new data ─────────
    def on_predict_clicked(self):
        """
        Open a small window where the user can enter ORIGINAL feature values.
        Now with proper range validation!
        """
        if self.best_model is None or not self.selected_features_current:
            messagebox.showerror("Error", "No trained model available.")
            return

        top = ctk.CTkToplevel(self)
        top.title("Predict New Sample")

        entry_widgets = {}      # numeric inputs
        dropdown_widgets = {}   # categorical inputs
        range_info = {}         # store min/max for validation

        row_idx = 0
        info_label = ctk.CTkLabel(
            top,
            text="Enter values for the original features used in training:",
            font=("Arial", 13, "bold")
        )
        info_label.grid(row=row_idx, column=0, columnspan=2,
                        padx=10, pady=(10, 5), sticky="w")
        row_idx += 1

        # Build inputs using ORIGINAL columns and dataset info
        for feat in self.selected_features_current:
            series = self.df[feat]  # original column

            # Label
            lbl_text = feat
            if series.dtype != object:
                # numeric -> show min/max hint AND store for validation
                try:
                    min_val = series.min()
                    max_val = series.max()
                    lbl_text += f" (min={min_val}, max={max_val})"
                    range_info[feat] = (min_val, max_val)  # STORE THIS
                except Exception:
                    range_info[feat] = (None, None)
            else:
                range_info[feat] = (None, None)

            lbl = ctk.CTkLabel(top, text=lbl_text + ":", anchor="w")
            lbl.grid(row=row_idx, column=0, padx=10, pady=3, sticky="w")

            # Decide: categorical vs numeric input
            if series.dtype == object or series.nunique() <= 20:
                # treat as categorical -> dropdown with unique values
                cat_values = [str(v) for v in series.dropna().unique()]
                if not cat_values:
                    cat_values = [""]
                var = ctk.StringVar(value=cat_values[0])
                opt = ctk.CTkOptionMenu(top, values=cat_values, variable=var, width=180)
                opt.grid(row=row_idx, column=1, padx=10, pady=3, sticky="w")
                dropdown_widgets[feat] = var
            else:
                # numeric -> free text entry
                ent = ctk.CTkEntry(top, width=180)
                ent.grid(row=row_idx, column=1, padx=10, pady=3, sticky="w")
                entry_widgets[feat] = ent

            row_idx += 1

        # Output label
        result_var = ctk.StringVar(value="Prediction: —")
        result_label = ctk.CTkLabel(top, textvariable=result_var, font=("Arial", 13, "bold"))
        result_label.grid(row=row_idx, column=0, columnspan=2,
                          padx=10, pady=(10, 5), sticky="w")
        row_idx += 1

        def do_predict():
            # 1) Build a one-row DataFrame with ORIGINAL feature values
            raw_dict = {}
            try:
                for feat in self.selected_features_current:
                    series = self.df[feat]
                    if feat in dropdown_widgets:  # categorical
                        raw_dict[feat] = dropdown_widgets[feat].get()
                    else:  # numeric
                        val_str = entry_widgets[feat].get().strip()
                        if val_str == "":
                            raise ValueError(f"Feature '{feat}' is empty.")
                        
                        # Convert to float
                        try:
                            val = float(val_str)
                        except ValueError:
                            raise ValueError(f"Feature '{feat}' must be a valid number.")
                        
                        # ===== VALIDATE RANGE =====
                        min_val, max_val = range_info[feat]
                        if min_val is not None and max_val is not None:
                            if val < min_val or val > max_val:
                                raise ValueError(
                                    f"Feature '{feat}' is out of valid range!\n"
                                    f"You entered: {val}\n"
                                    f"Valid range: [{min_val}, {max_val}]\n\n"
                                    f"Please enter a value within the training data range."
                                )
                        
                        raw_dict[feat] = val
                        
            except ValueError as e:
                messagebox.showerror("Input Error", str(e))
                return

            df_new_raw = pd.DataFrame([raw_dict])

            # 2) Reuse SAME feature engineering as training
            if self.df_for_fe is None or self.target_name_current is None:
                messagebox.showerror("Error", "Training metadata missing.")
                return

            df_concat = self.df_for_fe.copy()
            dummy = df_new_raw.copy()
            dummy[self.target_name_current] = np.nan  # target missing for new sample
            df_concat = pd.concat([df_concat, dummy], ignore_index=True)

            df_fe_all = feature_engineering(df_concat, self.target_name_current)
            X_all = df_fe_all.drop(columns=[self.target_name_current])

            # X for new sample is the last row
            X_new = X_all.tail(1)

            # Ensure same columns / order as model was trained on
            for col in self.trained_feature_names:
                if col not in X_new.columns:
                    X_new[col] = 0
            X_new = X_new[self.trained_feature_names]

            # 3) Predict
            try:
                pred = self.best_model.predict(X_new)[0]
            except Exception as e:
                messagebox.showerror("Prediction error", str(e))
                return

            # 4) Decode for classification
            if self.problem_type_current == "classification" and self.target_encoder is not None:
                try:
                    pred_decoded = self.target_encoder.inverse_transform([int(pred)])[0]
                    result_var.set(f"Prediction: {pred_decoded} (encoded={pred})")
                except Exception:
                    result_var.set(f"Prediction (encoded label): {pred}")
            else:
                try:
                    result_var.set(f"Prediction: {float(pred):.4f}")
                except Exception:
                    result_var.set(f"Prediction: {pred}")

        predict_btn = ctk.CTkButton(top, text="Predict", command=do_predict)
        predict_btn.grid(row=row_idx, column=0, columnspan=2,
                         padx=10, pady=(5, 10), sticky="ew")

    # ───────── Export button ─────────
    def on_export_clicked(self):
        messagebox.showinfo(
            "Export", "Export functionality not implemented yet 🙂")