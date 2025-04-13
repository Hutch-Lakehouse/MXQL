class MLViewManager:
    def __init__(self, engine):
        self.engine = engine
    
    def create_view(self, view_name, data):
        if isinstance(data, pd.DataFrame):
            data.to_sql(
                name=view_name,
                con=self.engine,
                schema="ML_Views",
                if_exists="replace",
                index=False
            )
        return view_name
