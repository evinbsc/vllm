import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.export import export
from torch.export import Dim, export

class DynamicShapesExample1(nn.Module):
    def forward(self, x):
        # Flatten the input tensor to match the expected input dimension 
        x = x.view(x.size(0), -1)
        
        x = self.layer1(x)
        x = self.layer2(x)

        return torch.relu(x)

    def __init__(self):
        super(DynamicShapesExample1, self).__init__()

        # Ensure layer dimensions are compatible 
        # This fixed mat1 and mat2 error
        self.layer1 = nn.Linear(6912, 5120)  
        self.layer2 = nn.Linear(5120, 3456) 

# For some reason I was running out of memory
# Used mixed Precision training for pytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inp1 = torch.randn(512, 6912).to(device).half()  

# Define dynamic dimensions
inp1_dim0 = Dim("inp1_dim0")

dynamic_shapes1 = {
    "x": {0: inp1_dim0},
}


model = DynamicShapesExample1().to(device).half()  

# To export the module with dynamic shapes
exported_module_example1 = export(model, (inp1,), dynamic_shapes=dynamic_shapes1)

# Save the exported program
torch.export.save(exported_module_example1, 'exported_program.pt2')



# class DynamicShapesExample1(nn.Module):
#     def forward(self, x):
#         # Flatten the input tensor to match the expected input dimension for nn.Linear
#         x = x.view(x.size(0), -1)
#         return torch.relu(x)

# inp1 = torch.randn(4096, 6912)

# # Define dynamic dimensions
# inp1_dim0 = Dim("inp1_dim0")
# inp1_dim1 = Dim("inp1_dim1", min=1000, max=8000)  # Adjust the range based on your requirements

# dynamic_shapes1 = {
#     "x": {0: inp1_dim0, 1: inp1_dim1},
# }

# # Export the module with dynamic shapes
# exported_module_example1 = export(DynamicShapesExample1(), (inp1,), dynamic_shapes=dynamic_shapes1)

# # Save the exported program
# torch.export.save(exported_module_example1, 'exported_program.pt2')




## EXPORTING PYTORCH MODULE
# class SiluAndMul(nn.Module):
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """PyTorch-native implementation equivalent to forward()."""
#         d = x.shape[-1] // 2
#         return F.silu(x[..., :d]) * x[..., d:]


# example_args = (torch.randn(4096, 6912),)

# act_fn = SiluAndMul()
# print(f"Result_vanilla = {act_fn(x=example_args[0])}")

# exported_program: torch.export.ExportedProgram = export(SiluAndMul(), args=example_args)
# print(exported_program)

# #Exporting to disk
# torch.export.save(exported_program, 'exported_program.pt2')




# LOAD FROM EXPORTED PROGRAM
#class SiluAndMulExported(nn.Module):
 #   def __init__(self):
  #      super().__init__()
   #     self.saved_exported_program = torch.export.load('exported_program.pt2').module()

    #def forward(self, x: torch.Tensor) -> torch.Tensor:
     #   return self.saved_exported_program(x)



# class SiluAndMulExported():
#     def __init__(self):
#         super().__init__()
#         self.saved_exported_program = torch.export.load('exported_program.pt2').module()

#     def __call__(self, x: torch.Tensor) -> torch.Tensor:
#         return self.saved_exported_program(x)


# act_fn = SiluAndMulExported()
# print(f"Result_exported = {act_fn(example_args[0])}")
