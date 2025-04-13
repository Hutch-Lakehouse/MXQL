from ipykernel.kernelbase import Kernel
from HuchML_core.parser import MxqlParser
from storage.views import MaterializedViewManager

class MxqlKernel(Kernel):
    def __init__(self):
        self.engine = get_engine()  # From config.database
        self.view_mgr = MaterializedViewManager(self.engine)
    
    def do_execute(self, code):
        # Unified parse -> transform -> transpile -> execute flow
        ast = MxqlParser().parse(code)
        py_code = Transpiler().generate(ast)
        exec(py_code, {"engine": self.engine})
        
        # Materialize results as SQL view
        if "predictions" in locals():
            self.view_mgr.create(locals()["predictions"])
