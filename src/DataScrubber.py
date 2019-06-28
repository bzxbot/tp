class DataScrubber:
    all_cols = ["idade","tp_sexo","tp_gestante","co_bairro_residencia","febre","mialgia","cefaleia","exantema","vomito","nausea","dor_costas","conjutivite","artrite","artralgia","petequia_n","leucopenia","laco","dor_retro","diabetes","hematolog","hepatopat","renal","hipertensao","acido_pept","auto_imune","res_chiks1","res_chiks2","resul_prnt","tp_result_exame","Tp_result_NS1","tp_result_isolamento","tp_result_rtpcr","tp_result_histopatologia","tp_result_imunohistoquimica","st_ocorreu_hospitalizacao","tp_autoctone_residencia","tp_classificacao_final","tp_criterio_confirmacao","tp_evolucao_caso", "tp_classificacao_final'"]
    X_columns = ["idade","tp_sexo","tp_gestante","co_bairro_residencia","febre","mialgia","cefaleia","exantema","vomito","nausea","dor_costas","conjutivite","artrite","artralgia","petequia_n","leucopenia","laco","dor_retro","diabetes","hematolog","hepatopat","renal","hipertensao","acido_pept","auto_imune","res_chiks1","res_chiks2","resul_prnt","tp_result_exame","Tp_result_NS1","tp_result_isolamento","tp_result_rtpcr","tp_result_histopatologia","tp_result_imunohistoquimica","st_ocorreu_hospitalizacao","tp_autoctone_residencia","tp_classificacao_final","tp_criterio_confirmacao","tp_evolucao_caso"]
    y_column = 'tp_classificacao_final'
    
    def scrub_data(self, data):
        X = data.filter(items=self.X_columns)
        X["tp_sexo"] = X["tp_sexo"].astype("category").cat.codes
        X["res_chiks2"].fillna(4, inplace=True)
        X["res_chiks2"].fillna(4, inplace=True)
        X["resul_prnt"].fillna(4, inplace=True)
        X["tp_result_exame"].fillna(4, inplace=True)
        X["Tp_result_NS1"].fillna(4, inplace=True)
        X["tp_result_isolamento"].fillna(4, inplace=True)
        X["tp_result_rtpcr"].fillna(4, inplace=True)
        X["tp_result_histopatologia"].fillna(4, inplace=True)
        X["tp_result_imunohistoquimica"].fillna(4, inplace=True)
        X["st_ocorreu_hospitalizacao"].fillna(9, inplace=True)
        X["diagnostico"] = X.apply(lambda row: 0 if row.tp_classificacao_final == 5 else 1, axis=1)
        print(X.info()) 
        X.dropna(inplace=True)
        print(X.info())
        y = X["diagnostico"] 
        X.drop(columns=["diagnostico", "tp_classificacao_final"], inplace=True)
        return X, y
