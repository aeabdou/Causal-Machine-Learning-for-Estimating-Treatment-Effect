def estimate_heterogeneity(df_main, y, population_1, population_2, n_splits, ml_model, binary_cols, all_y_list):
    """
    df: pandas.DataFrame
    y: str  (e.g., "own_value_ls")
    population_1, population_2: values in df['treatment'] to compare (pop1=1, pop2=0)
    n_splits: int
    ml_model: str or estimator (e.g., "catboost", "xgboost", "lightgbm", "rf", or a fitted-like object)
        Currently we have "catboost", "xgboost", "nn" options.
    all_y_list: list of strings that contains all the options for target variables, after passing the wanted target variable, the rest will be dropped 
    """

    #p_of_z = 1.0/2.0 #Delete later
    binary_cols = binary_cols

    df_population = df_main.copy() 
    # Defining the DataFrame:
    df_population = df_population[df_population["treatment"].isin([population_1, population_2])].copy()
    df_population["D"] = np.where(df_population["treatment"] == population_1, 1, 0)  # pop1=1, pop2=0

    # Defining p(Z):
    if {population_1, population_2} == {"Full cost", "Control"}:
        df_population.drop(columns=['p(Z)_half_vs_control','p(Z)_full_vs_half'], inplace=True)
        df_population["p(Z)"] = df_population['p(Z)_full_vs_control']

    elif {population_1, population_2} == {"Half cost", "Control"}:
        df_population.drop(columns=['p(Z)_full_vs_control','p(Z)_full_vs_half'], inplace=True)
        df_population["p(Z)"] = df_population['p(Z)_half_vs_control']

    elif {population_1, population_2} == {"Full cost", "Half cost"}:
        df_population.drop(columns=['p(Z)_full_vs_control','p(Z)_half_vs_control'], inplace=True)
        df_population["p(Z)"] = df_population['p(Z)_full_vs_half']


    # Dropping the treatment column:
    df_population.drop(columns=["treatment"], inplace=True)


    # Results container:
    results = {"blp": [], "gates": [], "clan": []}
    models_comparisons = {"small_lambda_blp":[], "small_lambda_gates":[]}  # to store the models comparison results. BLP and GATES small Lambdas

    
    # Dropping the columns that are not features:
    # list of all Y columns
    # Option A: copy + remove
    all_y_list = all_y_list.copy()
    others_y_list = all_y_list.copy()
    others_y_list.remove(y)              # y must be in the list
    df_population = df_population.drop(columns=others_y_list)

    # Creating a dictionary to store ML proxy predictions
    s_records = {idx: [] for idx in df_population.index} #dict with keys as the idx, and for each one, we have an empty list to each loop's prediction per idx/row.

    #-------------------------------------------------------------------------------------------------------------------

    # Big loop:
    base_seed = 7
    for split in range(n_splits):
        df = df_population.copy() # Reformating the original DF for each split so the split specific columns are re-defined

        # D-stratified 50/50 split → persistent 'fold' column
        # first assign all values to main
        df["fold"] = "main" 
        # Then assign half of each group of D (0 & 1) to aux randomly.
        aux_idx = (df.groupby("D", group_keys=False)
               .apply(lambda g: g.sample(frac=0.5, random_state= base_seed + split*10))).index   # group_keys=False
        df.loc[aux_idx, "fold"] = "aux"

    #-------------------------------------------------------------------------------------------------------------------
        # ML Proxy:
        # Defining the data for the two ML models:
        u_0_data = df[(df['fold'] == 'aux') & (df['D'] == 0)].copy()
        u_1_data = df[(df['fold'] == 'aux') & (df['D'] == 1)].copy()

        # 1) Define X, y for each aux subset
        drop_cols = [y, "D", "fold", "p(Z)"]

        X_0 = u_0_data.drop(columns=drop_cols)
        y_0 = u_0_data[y]

        X_1 = u_1_data.drop(columns=drop_cols)
        y_1 = u_1_data[y]

        # Decision the ML model to use
       # if ml_model == 'catboost':
        #    alpha = 0.45
        #    cb_params = dict(
        #        iterations=700,
        #        depth=8,
        #        learning_rate=0.06,
        #        loss_function=f"Quantile:alpha={alpha}",
        #        #loss_function="Huber:delta=2000",
        #        eval_metric=f"SMAPE",           # robust eval
        #        bootstrap_type="Bernoulli",
        #        subsample=0.8,
        #        rsm=0.8,
        #        l2_leaf_reg=6.0,
        #        od_type="Iter", od_wait=60,     # early stopping
        #        random_seed=39,
        #        verbose=False,
        #        allow_writing_files=False,
        #    )


#-----------------------------------------------------------------------

        if ml_model == 'catboost': 
            #alpha = 0.9
            cb_params = dict(
                iterations= 700, 
                depth= 8, 
                learning_rate= 0.05, 
               # loss_function= f"Quantile:alpha={alpha}", 
                #loss_function = 'RMSE',
                loss_function = 'Huber:delta=1000',
                #eval_metric= f"SMAPE", 
                #robust eval bootstrap_type="Bernoulli",
                #subsample=0.8, rsm=0.8, l2_leaf_reg=6.0,
                #od_type="Iter",
                #od_wait=60,
                # early stopping random_seed=39,
                verbose=False,
                # allow_writing_files=False,

        )

            # fit
            model_u1 = CatBoostRegressor(**cb_params).fit(Pool(X_1, y_1, cat_features=binary_cols))
            model_u0 = CatBoostRegressor(**cb_params).fit(Pool(X_0, y_0, cat_features=binary_cols))

        elif ml_model == 'xgboost':
            xgb_params = dict(
                n_estimators=700,
                max_depth=8,
                learning_rate=0.1,
                #subsample=0.8,
                #colsample_bytree=0.8,
                #random_state=39,
                #n_jobs=-1,
                objective="reg:squarederror",
                #enable_categorical=True
            )

            model_u1 = XGBRegressor(**xgb_params).fit(X_1, y_1)  # μ̂1(Z)
            model_u0 = XGBRegressor(**xgb_params).fit(X_0, y_0)  # μ̂0(Z)
            
        # Creating a RF option
        elif ml_model == 'rf':
            rf_params = dict(
                n_estimators=300,
                max_depth=8,
                max_features="sqrt",   # similar to colsample_bytree
                min_samples_leaf=1, # minimum number of samples required to be in a leaf node
                #n_jobs=-1, # ontrols how many CPU cores scikit-learn will use in parallel (n_jobs=-1: use all available cores.)
                #random_state=39
            )

            model_u1 = RandomForestRegressor(**rf_params).fit(X_1, y_1)  # μ̂1(Z)
            model_u0 = RandomForestRegressor(**rf_params).fit(X_0, y_0)  # μ̂0(Z)

        # Neural Network Option:
        elif ml_model == 'nn':
            # to float32
            X1 = np.asarray(X_1, dtype="float32"); y1 = np.asarray(y_1, dtype="float32")
            X0 = np.asarray(X_0, dtype="float32"); y0 = np.asarray(y_0, dtype="float32")

            model_u1 = keras.Sequential([
                layers.Input(shape=(X1.shape[1],)),
                layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
                #layers.BatchNormalization(),
                layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
                #layers.BatchNormalization(),
                layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
                #layers.BatchNormalization(),
                layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
                #layers.BatchNormalization(),
                layers.Dense(1)
                ])
            model_u1.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
                            loss="mse",
                            metrics=[keras.metrics.RootMeanSquaredError()]
                            )

            model_u0 = keras.Sequential([
                layers.Input(shape=(X0.shape[1],)),
                layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
                #layers.BatchNormalization(),
                layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
                #layers.BatchNormalization(),
                layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
                #layers.BatchNormalization(),
                layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
                #layers.BatchNormalization(),
                layers.Dense(1)
            ])
            model_u0.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
                            loss="mse",
                            metrics=[keras.metrics.RootMeanSquaredError()]
                            )

            # fit  (keep variables as the MODEL, not the History)
            model_u1.fit(X1, y1, epochs=50, batch_size=64, verbose=0)
            model_u0.fit(X0, y0, epochs=50, batch_size=64, verbose=0)

            ### Trying an Elastic Net Model option:
            # Elastic Net option:
        elif ml_model == 'elasticnet':
            enet_params = dict(
                l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],  # search over mix of L1/L2
                n_alphas=100,
                cv=5,
                random_state=39,
                max_iter=5000,
                n_jobs=-1
            )
            model_u1 = Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNetCV(**enet_params))
            ]).fit(X_1, y_1)
            model_u0 = Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNetCV(**enet_params))
            ]).fit(X_0, y_0)


        #elif ml_model == 'xyz': Place holder to add other model options later.

        #------------------------------------------------------------------------------------------------------


        # drop list you used when training
        drop_cols = [y, "D", "fold", "p(Z)"]

        # Applying the predictions to the main fold:
        # main fold
        main = df[df["fold"] == "main"].copy()

        # features for main (must match training: drop non-features)
        X_main = main.drop(columns=drop_cols).copy()


        if ml_model == 'nn':
            X_pred = X_main.reindex(columns=X_1.columns, fill_value=0).to_numpy(dtype="float32")
            model_u1_hat = np.asarray(model_u1.predict(X_pred)).reshape(-1)
            model_u0_hat = np.asarray(model_u0.predict(X_pred)).reshape(-1)
        else:
            # predict μ̂1 and μ̂0 on main, then S = μ̂1 − μ̂0 on any other model
            model_u1_hat = model_u1.predict(X_main)
            model_u0_hat = model_u0.predict(X_main)

        # we filter df by the location of the X_main, and then create a new column
        df.loc[main.index, "S(Z)"] = model_u1_hat - model_u0_hat

        # Also adding the base line preidictions B(Z):
        df.loc[main.index, "B(Z)"] = model_u0_hat


        # Collect S(Z) for rows that are in 'main' this split
        for idx, sval in df.loc[main.index, "S(Z)"].items():
            if pd.notnull(sval):
                s_records[idx].append(float(sval))

        #---------------------------------------------------------------------------------------------
        # BLP:

        # First Creating some BLP equation columns:
        # Creating the D-p column:
        mask = df['fold'].eq('main')
        df['D-p'] = np.nan
        #df.loc[mask, 'D-p'] = df.loc[mask, 'D'].astype(float) - float(p_of_z) 
        df.loc[mask, 'D-p'] = df.loc[mask, 'D'].astype(float) - df.loc[mask, 'p(Z)'].astype(float)


        # Create the shifted (centered) ML proxy on the MAIN sample
        mask = df['fold'].eq('main')
        df['S(Z)_shifted'] = np.nan
        mu_S = df.loc[mask, 'S(Z)'].mean()
        df.loc[mask, 'S(Z)_shifted'] = df.loc[mask, 'S(Z)'] - mu_S

        # Creating a single (D-p)*shifted_S_xgb column
        mask = df['fold'].eq('main')
        df['D-p_S(Z)_shifted'] = np.nan
        df.loc[mask, 'D-p_S(Z)_shifted'] = df.loc[mask, 'D-p'].astype(float) * df.loc[mask, 'S(Z)_shifted'].astype(float)


        # BLP Model:
        
        mask = df['fold'].eq('main')

        y_vec = df.loc[mask, y]  
        X = df.loc[mask, ['D-p', 'D-p_S(Z)_shifted', 'B(Z)', 'p(Z)']]

        X = sm.add_constant(X)  # adds the intercept α

        blp = sm.OLS(y_vec, X).fit(cov_type='HC1')  # robust SEs

        # Saving the BLP results:
        # We will save the R-Squared, R-Squared Adjusted, p-value, and the coefficients for D-p and  D-p_S(Z)_shifted.
        # column keys must match exactly how you named them in X


        beta2 = blp.params['D-p_S(Z)_shifted']      # slope on (D-p) * (S - mean_S)
        S_main = df.loc[mask, 'S(Z)'].to_numpy()    # raw S on the main split (not standardized)
        #var_y_main = float(np.var(df.loc[mask,y], ddof=0)) # population var to divide by

        small_lambda_blp = float(beta2**2) * float(np.var(S_main, ddof=0)) 

        #small_lambda_blp = float(beta2**2) * float(np.var(S_main, ddof=0))  / var_y_main


        # inside the loop, AFTER fitting `blp`
        k1, k2 = "D-p", "D-p_S(Z)_shifted"

        def grab(key):
            lo, hi = blp.conf_int().loc[key]
            return {
                "name": key,
                "coef": blp.params[key],
                "std_err": blp.bse[key],
                "z": blp.tvalues[key],
                "p": blp.pvalues[key],
                "ci_low": lo,
                "ci_high": hi,
            }

        split_blp = [
            {"name": "D_minus_p", **grab(k1)},
            {"name": "D_minus_p_times_S", **grab(k2)},
        ]

        beta2 = blp.params['D-p_S(Z)_shifted']      # slope on (D-p) * (S - mean_S)
        S_main = df.loc[mask, 'S(Z)'].to_numpy()    # raw S on the main split (not standardized)

        #small_lambda_blp = float(beta2**2) * float(np.var(S_main, ddof=0))  # population var
        small_lambda_blp = float(beta2**2) * float(np.var(S_main, ddof=1))


        results["blp"].append(split_blp)   
        models_comparisons["small_lambda_blp"].append(small_lambda_blp)
        
    #return results # Only used for debugging
        

       #-----------------------------------------------------------------------------------------------
       # GATES

       # We will start by creating a new column that assigns quantile groups to each S(Z) value.
       # MAIN-only groups for GATES
        mask_main = df["fold"].eq("main")
        s_main = df.loc[mask_main, "S(Z)"].dropna()

        df["Group"] = np.nan
        df.loc[mask_main, "Group"] = (
            pd.qcut(s_main, q=5, labels=[1,2,3,4,5], duplicates="drop")
            .astype(float)
        )

        # Creating the main column for the GATES model, which is T_K
        # T_k = (D - p) * 1{Group=k}
        for k in range(1, 6):
            df.loc[mask_main, f"T{k}"] = (
                df.loc[mask_main, "D-p"] *
                (df.loc[mask_main, "Group"].astype(int) == k).astype(int)
            )


        # Fit GATES Model:
        dfm = df.loc[mask_main, [y, "T1","T2","T3","T4","T5","B(Z)"]].dropna()
        X = sm.add_constant(dfm[["T1","T2","T3","T4","T5","B(Z)"]], has_constant="add")
        gates = sm.OLS(dfm[y], X).fit(cov_type="HC1")
        params = gates.params


        # Saving the GATES results in our results dictionary:
        ci = gates.conf_int()  # 95% by default
        params = gates.params
        bse = gates.bse

        split_gates = []
        for k in range(1, 6):
            tk = f"T{k}"
            split_gates.append({
                "group": k,
                "gamma": float(params[tk]),
                "std_err": float(bse[tk]),
                "ci_low": float(ci.loc[tk, 0]),
                "ci_high": float(ci.loc[tk, 1]),
            })

        results["gates"].append(split_gates)

        bar_lambda = 0
        for k in range(1, 6):
            tk = f"T{k}"
            bar_lambda += float(gates.params[tk])**2 
        
        #bar_lambda_gates = (bar_lambda / 5.0) / var_y_main
        bar_lambda_gates = (bar_lambda / 5.0) 


        models_comparisons["small_lambda_gates"].append(bar_lambda_gates)

        

       #-----------------------------------------------------------------------------------------------
       # CLAN
        # Steps:
            #1. Get only the numeric columns.
            #2. Remove all the columns created in the previous steps and the target column.
            #3. Remove all the columns that numericly coded but are binary.
            #4. Remove additional columns that are categorical but were not detected as binary (found after in the CLAN table)
            #5. Getting a list of all columns that will be entered to CLAN

        # 1. Keeping only numeric columns
        df_numeric = df.select_dtypes(include=[np.number])

        # 2. Removing the columns created in the previous steps
        clan_exclude = [
            "D",
            "S(Z)",
            "D-p",
            "S(Z)_shifted",
            "D-p_S(Z)_shifted",
            "B(Z)",
            "p(Z)",
            "Group",
            "T1", "T2", "T3", "T4","T5",
            y
        ]
        # 3. Removing binary columns:
        binary_cols_01 = [c for c in df_numeric.columns
                        if set(df_numeric[c].dropna().unique()) == {0, 1}]

        #4.Remove additional columns that are categorical but were not detected as binary (found after in the CLAN table)
        additional_features_drop = ['vehicle_motorcycle_e','vehicle_toktok_e','vehicle_tricycle_e','skip_meal_e','has_heater_e','has_fridge_e','cart_e','bicycle_e']

        #5. Getting a list of all columns that will be entered to CLAN
        # Combing all exclude column lists and filter the numeric DF:
        cols_to_remove = clan_exclude + binary_cols_01 + additional_features_drop

        # filtering the numeric DF to get only a DF with only useful columns
        df_clan_cols = df_numeric.drop(columns=cols_to_remove)

        # This is our final feature list
        clan_cols = df_clan_cols.columns

        # Creating a Clan Table
        # 1. Filter to inlcude only the main fold
        # 2. Include only the CLAN columns + the group column
        # 3. Include only group 1 and 5

        # filtering to include only main fold
        df_main = df[df['fold'] == 'main']

        # Filtering to include only CLAN columns + group
        df_clan = df_main.loc[:, list(clan_cols) + ["Group"]].copy()

        # keep only groups 1 and 5 (fixing the var name)
        df_clan = df_clan[df_clan["Group"].isin([1, 5])]



        #  Calculating the means, the diff, the SE, the z, the p, the CI for each variable
        stats = (
            df_clan.groupby("Group")[clan_cols]
            .agg(["mean", "std", "count"])
            .reindex([1, 5])  # ensure order: G1, G5
        )

        rows = []
        zcrit = 1.96  # 95% CI using normal critical value

        for col in clan_cols:
            m1 = stats.loc[1, (col, "mean")]
            m5 = stats.loc[5, (col, "mean")]
            s1 = stats.loc[1, (col, "std")]
            s5 = stats.loc[5, (col, "std")]
            n1 = stats.loc[1, (col, "count")]
            n5 = stats.loc[5, (col, "count")]

            diff = m5 - m1
            
            # Welch SE for difference in means
            se = np.sqrt((s5**2) / n5 + (s1**2) / n1) 
            z  = diff / se

            # Two-sided p-value from the normal CDF via erf
            # Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
            
            p_two_sided = 2 * (1 - (0.5 * (1 + erf(abs(z) / sqrt(2)))))
        

            ci_lo = diff - zcrit * se 
            ci_hi = diff + zcrit * se 

            # save the results to our results dictionary
            rows.append({
                "covariate": col,
                "Mean G1 (predicted least affected)": m1,
                "Mean G5 (predicted most affected)": m5,
                "Diff (G5 - G1)": diff,
                "SE (Welch)": se,
                "z": z,
                "p (two-sided)": p_two_sided,
                "CI 95% lower": ci_lo,
                "CI 95% upper": ci_hi,
            })

        # save CLAN for this split (one big list of covariate rows)
        results["clan"].append(rows)

        # End of the loop

        # Calculating the median S(Z) for each row/list
        s_median_series = pd.Series(
            {idx: (np.median(vals) if len(vals) > 0 else np.nan) for idx, vals in s_records.items()}
        ).reindex(df_population.index)

        # If you specifically want a plain list:
        s_medians_per_row = s_median_series.tolist()



    #------------------------------------------------------------------------------------------------
    return results, models_comparisons, s_medians_per_row # the final result dictionaries
