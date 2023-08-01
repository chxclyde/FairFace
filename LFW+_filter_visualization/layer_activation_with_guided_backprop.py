"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch.nn as nn
import os
import torch
from torch.nn import ReLU
from torchvision import models
from misc_functions import (my_get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, target_layer_name = None):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.hooks = []
        
        self.target_layer_out = None
        self.target_layer_name = target_layer_name
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()
        

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.children())[0]
        self.hooks.append(first_layer.register_backward_hook(hook_function))

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        
        for name, module in self.model.named_modules():
            if isinstance(module, ReLU):
                if name == self.target_layer_name:
                    def target_forward_hook_function(module, ten_in, ten_out):
                        """
                        Store results of forward pass
                        """
                        self.target_layer_out = ten_out 
                        self.forward_relu_outputs.append(ten_out)
                    self.hooks.append(module.register_backward_hook(relu_backward_hook_function))
                    self.hooks.append(module.register_forward_hook(target_forward_hook_function))
                    break
                self.hooks.append(module.register_backward_hook(relu_backward_hook_function))
                self.hooks.append(module.register_forward_hook(relu_forward_hook_function))

    def generate_gradients(self, input_image, filter_pos):
        self.model.zero_grad()
        # Forward pass
        x = input_image
        self.model(x)
        conv_output = torch.sum(torch.abs(self.target_layer_out[0, filter_pos]))
        # Backward pass
        conv_output.backward()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr
    def remove_hooks ( self):
        """remove all the hooks"""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return image_paths

def visualize_target_filter( prep_img,pretrained_model,target_layer_name , filter_pos, file_name_to_export):
    filter_dir = target_layer_name+ "_"+"filter-" + str(filter_pos) 
    GBP = GuidedBackprop(pretrained_model, target_layer_name)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, filter_pos)
    # Save colored gradients
    save_gradient_images(filter_dir,guided_grads, file_name_to_export + '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(filter_dir,grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(filter_dir,pos_sal, file_name_to_export + '_pos_sal')
    save_gradient_images(filter_dir,neg_sal, file_name_to_export + '_neg_sal')
    print('Layer Guided backprop completed')
    GBP.remove_hooks()

def get_results():
    pretrained_model = models.resnet34(pretrained=True)
    pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, 6)
    device = torch.device('cuda:1')
    pretrained_model.load_state_dict(torch.load("/home/clydechx/FairFace/LFWA+_model.pt",map_location=device))
    dataset_dir= '/home/clydechx/FairFace/mydataset/lfw/lfw'
    image_paths = get_image_paths(dataset_dir)
    to_visual = {'layer1.2.relu': [39], 'layer2.1.relu': [1], 'layer2.2.relu': [60, 105], 'layer2.3.relu': [105], 'layer3.0.relu': [7, 22, 144, 156, 231, 241, 251], 'layer3.1.relu': [18, 111, 118, 144, 146, 151, 162, 180, 197, 205, 208, 213, 229, 231, 236, 251], 'layer3.2.relu': [18, 118, 130, 151, 162, 179, 200, 205, 225, 241], 'layer3.3.relu': [118, 141, 156, 162, 200, 205], 'layer3.4.relu': [118, 134, 205, 251]}
    for image_path in image_paths :
        for layer_name,filters in to_visual.items():
            for filter in filters:
                file_name = image_path.split("/")[-1].replace('.jpg' , '') +"_" + layer_name + "_" + "filter-"+str(filter)
                prep_img = my_get_example_params(image_path)
                visualize_target_filter(prep_img, pretrained_model ,  layer_name , filter,file_name)
                
get_results()

# lfw+ all dataset activation out layer
# relu {}
# layer1.0.relu {}
# layer1.1.relu {}
# layer1.2.relu {39: 0.028117582618221653}
# layer2.0.relu {}
# layer2.1.relu {1: 0.011411356582070476}
# layer2.2.relu {60: 0.013511046193171445, 105: 0.01314588278254519}
# layer2.3.relu {105: 0.034873105714807374}
# layer3.0.relu {7: 0.012415555961292679, 22: 0.023096585722110645, 144: 0.013237173635201752, 156: 0.01031586635019171, 231: 0.08864341792952346, 241: 0.05513967500456454, 251: 0.02300529486945408}
# layer3.1.relu {18: 0.020996896111009678, 111: 0.08188789483293774, 118: 0.07750593390542268, 144: 0.07960562351652364, 146: 0.15775059339054226, 151: 0.021179477816322803, 162: 0.027204674091656018, 180: 0.23114843892641956, 197: 0.3521088186963666, 205: 0.012598137666605806, 208: 0.01077232061347453, 213: 0.04035055687420121, 229: 0.016341062625524923, 231: 0.05723936461566551, 236: 0.10744933357677561, 251: 0.16359320796056234}
# layer3.2.relu {18: 0.02683951068102976, 118: 0.09905057513237174, 130: 0.013054591929888625, 151: 0.025287566185868176, 162: 0.014697827277706774, 179: 0.010589738908161402, 200: 0.010863611466131094, 205: 0.05632645608909987, 225: 0.05970421763739273, 241: 0.02756983750228227}
# layer3.3.relu {118: 0.0383421581157568, 141: 0.01652364433083805, 156: 0.05970421763739273, 162: 0.011046193171444221, 200: 0.06855943034507943, 205: 0.023279167427423773}
# layer3.4.relu {118: 0.026656928975716632, 134: 0.01077232061347453, 205: 0.024922402775241922, 251: 0.01077232061347453}
# layer3.5.relu {}
# layer4.0.relu {}
# layer4.1.relu {}
# layer4.2.relu {}

# if __name__ == '__main__':
#     target_layer_name = "layer1.1.relu"
#     filter_pos = 5
#     target_example = 2  # Spider
#     (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
#         get_example_params(target_example)
#     # pretrained_model = models.alexnet(pretrained=True)
#     print("file_name",file_name_to_export)
#     # File export name
#     file_name_to_export = file_name_to_export + '_layer' + str(target_layer_name) + '_filter' + str(filter_pos)
#     # Guided backprop
#     GBP = GuidedBackprop(pretrained_model, target_layer_name)
#     # Get gradients
#     guided_grads = GBP.generate_gradients(prep_img, filter_pos)
#     # Save colored gradients
#     save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color')
#     # Convert to grayscale
#     grayscale_guided_grads = convert_to_grayscale(guided_grads)
#     # Save grayscale gradients
#     save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
#     # Positive and negative saliency maps
#     pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
#     save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
#     save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
#     print('Layer Guided backprop completed')
