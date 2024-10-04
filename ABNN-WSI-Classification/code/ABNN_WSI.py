import argparse
import numpy as np
from TensorDataset import *
from torch.utils.data.sampler import SubsetRandomSampler
from modelsMinMax import *
import torch.optim as optim
import random
import torchvision
from sklearn.metrics import *
import pdb
import time
import subprocess
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

def get_gpu_usage():
    result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "-format=csv,nounits,noheader"], stdout=subprocess.PIPE)
    print(result.stdout.decode("utf-8"))

#Loaders for datasets
def loaders(data_dir, val_dir, test_dir,batch_size=1,random_seed=0,shuffle=True, extension='svs'):

    dataset = TensorDataset(data_dir,extension) #train dataset
    dataset_val= TensorDataset(val_dir, extension) #validation dataset
    dataset_test = TensorDataset(test_dir, extension) #test dataset


    num_sample = len(dataset) #number of samples in train set
    num_classes = len(dataset.classes) #number of classes
    indices = list(range(num_sample)) #indices of samples in train set

    num_sample_val = len(dataset_val) #number of samples in validation set
    indices_val=list(range(num_sample_val)) #indices of samples in validation set

    num_sample_test = len(dataset_test) #number of samples in test set
    indices_test = list(range(num_sample_test)) #indices of samples in test set

    #shuffle of the samples
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        np.random.shuffle(indices_val)
        np.random.shuffle(indices_test)

    train_sampler = SubsetRandomSampler(indices) #Samples elements randomly in train set
    valid_sampler = SubsetRandomSampler(indices_val) #Samples elements randomly in validation set
    test_sampler = SubsetRandomSampler(indices_test) #Samples elements randomly in test set

    #train dataset loader
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
    )

    # validation dataset loader
    valid_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, sampler=valid_sampler
    )

    # test dataset loader
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler
    )


    return (num_classes,train_loader,valid_loader, test_loader)



#Creation of 3D tensors starting from images
def tensors_creation(model_image, args, device):
    #load Image Dataset
    dset = ImageDataset(root_dir=args.data_dir, patch=args.patch_size, scale=args.patch_scale, overlap=0,
                        device=device, extension=args.ext)
    #for each image
    for i in range(len(dset)):
        try:
            #return image, label and path of the image
            input, label, filename = dset.__getitem__(i)
            print(filename)
            print(label)
            #number of map filters for the output tensor
            num_filters=args.filters_in
            #height and width of the tensor (height image/patch_size and width image/patch_size)
            H = input.shape[0]
            W = input.shape[1]
            print(W,H)
            #path of the destination tensor file
            file_dst = os.path.join(args.save_dir, filename) + ".pth"
            if not os.path.exists(file_dst):
                control_param=3 #parameter to control the final dimension of the sensor, which has to be at least a dimension of 3x3xnum_filters
                start_H=0
                start_W=0
                #control of tensor size
                if (H < control_param and W < control_param):
                    tensor_U = torch.zeros([control_param, control_param, num_filters], device=device)
                    start_H = 1
                    start_W = 1
                else:
                    if (H<control_param or W<control_param):
                        if H<control_param:
                            tensor_U = torch.zeros([control_param, W, num_filters], device=device)
                            start_H=1
                        if W<control_param:
                            tensor_U = torch.zeros([H, control_param, num_filters], device=device)
                            start_W=1
                    else:
                        tensor_U = torch.zeros([H, W, num_filters], device=device)
                #batch size for the model equal to width of the tensor
                bs = W
                for h in range(0, H):
                    for w in range(0, W, bs):
                        dim = bs
                        if w > W - bs:
                            dim = W - bs
                        batch = input[h][w:w + dim][:][:]
                        #pass the batch of patches to the model
                        ris = model_image(batch.cuda(device=device))
                        ris = torch.squeeze(ris)
                        #insertion of the results of the batch in the final tensor
                        tensor_U[h+start_H, w+start_W:w+start_W + dim] = ris.detach()
                #addition of two new dimension to the tensor for train and test phases
                tensor_U = tensor_U.unsqueeze(0)
                tensor_U = tensor_U.unsqueeze(0)
                #tensor_U.requires_grad_(True)
                if not os.path.exists(file_dst):
                    #saving of the tensor in the destination path
                    torch.save(tensor_U, file_dst)
                    #get_gpu_usage()
                print(torch.cuda.memory_allocated())
                print(torch.cuda.memory_reserved())
                torch.cuda.empty_cache()
                del tensor_U
                #get_gpu_usage()
                time.sleep(5)
        except Exception as ex:
            print(ex)
            continue


def save_attention_maps(max_attention_map, min_attention_map, save_name):
    max_attention_map_np = max_attention_map[0, 0].detach().cpu().numpy()  # Select first filter from batch
    min_attention_map_np = min_attention_map[0, 0].detach().cpu().numpy()  # Same for min
    # Plot the max attention map
    fig1, ax1 = plt.subplots()  # Create a figure (fig) and axes (ax)
    sns.heatmap(max_attention_map_np.squeeze(0), cmap='viridis', ax=ax1)  # Specify ax=ax1 to use the created axes
    ax1.set_title('Max-Pooling Attention Map')  # Use set_title() to set the title
    fig1.savefig(RESULTS_DIR+ "/Attention_heatmaps/"+save_name+"_max"+'.png', dpi=300, bbox_inches='tight')

    # Plot the min attention map
    fig2, ax2 = plt.subplots()  # Create a figure (fig) and axes (ax)
    sns.heatmap(min_attention_map_np.squeeze(0), cmap='viridis', ax=ax2)  # Specify ax=ax2 to use the created axes
    ax2.set_title('Min-Pooling Attention Map')  # Use set_title() to set the title
    fig2.savefig(RESULTS_DIR+"/Attention_heatmaps/"+save_name+"_min"+'.png', dpi=300, bbox_inches='tight')
 


def overlay_attention_map(input_tensor,max_attention_map,save_name ):
    max_attention_map_np = max_attention_map[0, 0].detach().cpu().numpy()  # Select first filter from batch
    #pdb.set_trace()
    # Assuming 'input_image' is a 2D numpy array representing your input image
    input_image = input_tensor[0, 0][0].detach().cpu().numpy()  # Example: Grayscale image from input tensor
    print(input_image.shape)
   
    #input_image  = input_image
    # Resize the attention map to match the input image size (if necessary)
    #attention_map_resized = cv2.resize(max_attention_map_np, (input_image.shape[1], input_image.shape[0]))

    # Normalize attention map for visualization
    attention_map_resized = max_attention_map_np.squeeze(0)
    #attention_map_resized = (attention_map_resized - np.min(attention_map_resized)) / np.ptp(attention_map_resized)
    print(attention_map_resized.shape)
    # Overlay attention map on input image
    fig3, ax3 = plt.subplots()
    # Show the original image in grayscale
    ax3.imshow(input_image, cmap='gray', alpha=0.7)
    # Optionally overlay the attention map
    ax3.imshow(attention_map_resized, cmap='jet', alpha=0.3)  # Uncomment if needed
    # Set the title correctly using set_title()
    ax3.set_title('Max Attention Map Overlay')
    # Save the figure
    fig3.savefig(RESULTS_DIR+"/Attention_heatmaps/"+save_name+"_overlay_max"+'.png', dpi=300, bbox_inches='tight')



# Grad-CAM function
def grad_cam(model, output, M_mat, m_mat, input_image, target_class,save_name):
    #model.eval()
    # Forward pass
    #output, M_mat, m_mat = model(input_image)
    output = output[target_class.cpu().numpy()[0]]  # Get output for the target class
    #pdb.set_trace()
    # Backward pass
    model.zero_grad()
    M_mat.retain_grad()
    m_mat.retain_grad()
    
    output.backward(retain_graph=True)

    # Get gradients with respect to attention maps
    M_mat_grad = M_mat.grad
    m_mat_grad = m_mat.grad

    # Compute weighted combination of gradients and attention maps
    M_mat_cam = torch.mean(M_mat_grad, dim=(2, 3), keepdim=True) * M_mat
    m_mat_cam = torch.mean(m_mat_grad, dim=(2, 3), keepdim=True) * m_mat

    # Sum over channels to get the final Grad-CAM heatmap
    grad_cam_map = M_mat_cam.sum(dim=1) + m_mat_cam.sum(dim=1)
    
    fig, ax = plt.subplots()
    # Plot the Grad-CAM heatmap on the axis
    heatmap_data = grad_cam_map.squeeze().detach().cpu().numpy()
    im = ax.imshow(heatmap_data, cmap='jet')

    # Set the title for the axis
    ax.set_title("Grad-CAM Heatmap")

    # Optionally add a colorbar
    fig.colorbar(im, ax=ax)
    fig.savefig(RESULTS_DIR+"/GradCam_maps/"+save_name+'.png', dpi=300, bbox_inches='tight')
    return grad_cam_map



# Ablation-CAM function
def ablation_cam(model, output, M_mat, m_mat, input_image, target_class, save_name, patch_size=3):
    # Ensure model is in evaluation mode
    #pdb.set_trace()
    model.eval()
    
    # Extract the output for the target class
    output = output[target_class.cpu().numpy()[0]]

    # Initialize gradients
    #model.zero_grad()
    
    # Prepare for ablation: copy the attention maps (max and min)
    M_mat_cam = M_mat.clone().detach()
    m_mat_cam = m_mat.clone().detach()
    
    # Create a map to store the ablation results
    ablation_map = torch.zeros_like(M_mat_cam)
    
    # Loop through the attention map (M_mat) and ablate patches
    for i in range(0, M_mat_cam.size(2), patch_size):
        for j in range(0, M_mat_cam.size(3), patch_size):
            # Create a copy of the attention maps with one patch ablated
            ablated_M_mat = M_mat_cam.clone()
            ablated_M_mat[:, :, i:i+patch_size, j:j+patch_size] = 0  # Zero out a patch
            
            # Forward pass with the ablated attention map
            with torch.no_grad():
                ablated_output, _, _ = model(input_image)
                ablated_output = ablated_output[target_class.cpu().numpy()[0]]
            
            # Compute the change in the target class score
            score_change = output - ablated_output
            #print(score_change)
            # Store the score change in the ablation map
            ablation_map[:, :, i:i+patch_size, j:j+patch_size] = score_change
    """
    # Repeat the same process for the min attention map if desired (you can also skip if focusing on max only)
    for i in range(0, m_mat_cam.size(2), patch_size):
        for j in range(0, m_mat_cam.size(3), patch_size):
            # Create a copy of the min attention map with one patch ablated
            ablated_m_mat = m_mat_cam.clone()
            ablated_m_mat[:, :, i:i+patch_size, j:j+patch_size] = 0  # Zero out a patch
            
            # Forward pass with the ablated min attention map
            with torch.no_grad():
                ablated_output, _, _ = model(input_image)
                ablated_output = ablated_output[target_class.cpu().numpy()[0]]
            
            # Compute the change in the target class score
            score_change = output - ablated_output
            
            # Store the score change in the ablation map
            ablation_map[:, :, i:i+patch_size, j:j+patch_size] += score_change
    """
    # Sum over channels to get the final Ablation-CAM heatmap
    ablation_cam_map = ablation_map.sum(dim=1)

    # Plot the Ablation-CAM heatmap
    fig, ax = plt.subplots()
    heatmap_data = ablation_cam_map.squeeze().detach().cpu().numpy()
    im = ax.imshow(heatmap_data, cmap='jet')

    # Set the title for the axis
    ax.set_title("Ablation-CAM Heatmap")

    # Optionally add a colorbar
    fig.colorbar(im, ax=ax)

    # Save the heatmap as an image file
    fig.savefig(RESULTS_DIR + "/AblationCam_maps/"+save_name+'.png', dpi=300, bbox_inches='tight')

    return ablation_cam_map





def save_high_attention_crops(grad_cam_map,tiled_dict,folder,filename, thresh=0.0075):
    #tiled_dict, label, filename = dset_test.__getitem__(k)
    indices = torch.nonzero(grad_cam_map > thresh)
    for j in range(len(indices)):
        ind_1, ind_2 = indices[j][2].cpu().numpy(), indices[j][3].cpu().numpy()
        tensor = tiled_dict[ind_1,ind_2 ]
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        tensor = tensor.permute(1, 2, 0)
        image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_np)
        if not os.path.exists(RESULTS_DIR+"/"+folder+"/"+filename):
            os.makedirs(RESULTS_DIR+"/"+folder+"/"+filename)
        image.save(RESULTS_DIR+"/"+folder+"/"+filename+"/"+str(ind_1)+"_"+str(ind_2)+".png")




#Test of the model
def test(model,test_loader,device, dset, save_attention_results=False):
    model.eval()
    true_labels = []
    predicted_labels = []
    classes, class_to_idx = dset.find_classes()
    #for each test/validation tensor
    for i in range(len(test_loader)):
        #index of the tensor
        k = test_loader.sampler.indices[i]
        
        try:
            #with torch.no_grad():
            #recover tensor information
            ris_model, label, file_name = dset.tensor_and_info(k)
            #ris_model.requires_grad = True
            #print(file_name)
            #permutation of the tensor in order to fit in the model
            ris_model = ris_model.permute(0, 1, 4, 2, 3)
            
            #output label prediction
            output, max_attention_map, min_attention_map = model(ris_model.cuda(device=device))
            
            
            if save_attention_results==True:
                
                # Save attention maps - max and min
                save_attention_maps(max_attention_map, min_attention_map, file_name.split("/")[-1]+"_"+str(k))
                # Overlay attention maps on downscaled image
                overlay_attention_map(ris_model,max_attention_map, file_name.split("/")[-1]+"_"+str(k))
                # Create grad cam map
                grad_cam_map = grad_cam(model,output, max_attention_map, min_attention_map, ris_model, label, file_name.split("/")[-1])
                # Get the original tile tensors
                tiled_dict, label, filename = dset.__getitem__(k)
                # Save high attention grad cam crops
                save_high_attention_crops(grad_cam_map,tiled_dict,"GradCam_high_attention_image_crops", file_name.split("/")[-1],0.0075)
                #pdb.set_trace()
                # Save high attention ablation cam crops    
                ablation_cam_map = ablation_cam(model, output, max_attention_map, min_attention_map, ris_model, label, file_name.split("/")[-1], patch_size=3)
                save_high_attention_crops(ablation_cam_map,tiled_dict,"AblationCam_high_attention_image_crops", file_name.split("/")[-1],0.0005)
                
                
            output = output.unsqueeze(0)
            #comparison between true label and predicted label
            true_labels.append(label.cpu().numpy()[0])
            predicted_labels.append(np.argmax(output.detach().cpu().numpy()[0]))


        except Exception as ex:
            print(ex)
        continue
    #measure of the performance
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels)
    conf_mat = confusion_matrix(true_labels, predicted_labels)
    evaluation_metrics = pd.DataFrame({"classes":classes, "precision":precision, "recall":recall,"f1-score":f1, "support":support})
    evaluation_metrics.to_csv(RESULTS_DIR+"/Evaluation_metrics/evaluation_metric.csv")
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=classes)
    disp.plot()
    # Save the plot
    plt.savefig(RESULTS_DIR+"/Evaluation_metrics/confusion_matrix.png") 
    return accuracy, precision, recall, f1, conf_mat

#Training of the model
def train(model, args, device, optimizer, num_epochs, train_loader, valid_loader, test_loader, model_path, model_path_fin, bs):
    #recover of the all datasets
    dset = ImageDataset(root_dir=args.data_dir, patch=args.patch_size, scale=args.patch_scale, overlap=0,
                        device=device, extension=args.ext)
    dset_test = ImageDataset(root_dir=args.test_dir, patch=args.patch_size, scale=args.patch_scale,
                             overlap=0,
                             device=device, extension=args.ext)
    dset_val = ImageDataset(root_dir=args.val_dir, patch=args.patch_size, scale=args.patch_scale, overlap=0,
                            device=device, extension=args.ext)
    #augmentation datasets
    dset_aug = ImageDataset(root_dir=args.aug_dir, patch=args.patch_size, scale=args.patch_scale, overlap=0,
                            device=device, extension=args.ext)
    dset_aug2 = ImageDataset(root_dir=args.aug_dir2, patch=args.patch_size, scale=args.patch_scale, overlap=0,
                            device=device, extension=args.ext)
    classes, class_to_idx = dset.find_classes()
    print(class_to_idx)
    
    best_epoch = 0
    loss = torch.nn.CrossEntropyLoss()
    #parameter of control for the saving of the final model
    mean_f1=0

    for epoch in range(num_epochs):
        running_samples = 0
        running_losses = 0
        model= (model.train(True))
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        accuracy = 0
        #number of augmentation
        num_op = 12
        global array_aug
        array_aug = []
        #array_aug is the array to store the augmentation already used
        for j in range(len(dset)):
            column = []
            for i in range(num_op):
                column.append(i)
            array_aug.append(column)

        for i in range(num_op):
            #train for a single operation of augmentation
            true_labels, predicted_labels, running_loss, running_sample = train_single_op(model,
                                                                                      train_loader,
                                                                                      bs, optimizer, loss,
                                                                                      dset, device,dset_aug,dset_aug2)
            #calculation of performance for single operation
            running_samples += running_sample
            running_losses += running_loss
            accuracy += accuracy_score(true_labels, predicted_labels)
        print("Actual model obtained at epoch {}, Accuracy={},Loss={}".format(
            str(epoch), accuracy / num_op, running_loss / running_samples))

        #save the current trained model
        model_string = model_path_fin + ".pt"
        #torch.save(model.state_dict(), model_string)
        #validate the model
        accuracy, precision, recall, F1, conf_mat = test(model, valid_loader, device, dset_val)
        print(
            "Actual model obtained at epoch {}, Validation/Accuracy={}, Mean(F1)={}, CONF={}".format(
                str(epoch), accuracy, np.mean(F1), conf_mat))

        #if mean of F1 (for validation set) is higher than F1 previously computated, save the new model
        #if (np.mean(F1)) >=mean_f1:
        if accuracy >=mean_f1:
            model_string = model_path_fin + ".pt"
           # model_string = model_path+ + ".pt"
            #torch.save(model.state_dict(), model_string)
            torch.save(model, model_string)
            #mean_f1 = np.mean(F1)
            mean_f1 = accuracy
            best_epoch = epoch
            print("MODEL SAVED!!! Actual best model obtained at epoch {}, Accuracy={}, Mean(F1)={}".format(
                str(best_epoch), accuracy, np.mean(F1)))
        #test the model
        accuracy, precision, recall, F1, conf_mat = test(model, test_loader, device,
                                                                     dset_test)

        print("Actual model obtained at epoch {}, Test/Accuracy={} Mean(F1)={},CONF={}".format(
            str(epoch), accuracy, np.mean(F1), conf_mat))

    #return best_epoch, np.mean(F1)
    return best_epoch, accuracy

#Train a single augmentation operation
def train_single_op(model, train_loader, bs, optimizer, loss, dset, device,dset_aug,dset_aug2):
    true_labels = []
    predicted_labels = []
    running_loss = 0.000000001
    running_samples = 0
    j = 0
    outputs = None
    labels = None
    optimizer.zero_grad()
    shift = 3
    for i in range(len(train_loader)):
        k = train_loader.sampler.indices[i]
        j += 1
        try:
            ris_model, label, file_name = dset.tensor_and_info(k)
            #ris_model.requires_grad=True
            flag = 0
            num_op = 12
            operation = np.random.choice(num_op)
            while flag == 0:
                if array_aug[k][operation] < num_op + 1:
                    array_aug[k][operation] = num_op + 1
                    flag = 1
                else:
                    operation = np.random.choice(num_op)
            # rotation of 90° and flip along axis 1
            if operation == 1:
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = ris_model.transpose(0, 1).flip(1)
                ris_model = ris_model.unsqueeze(0)
                ris_model = ris_model.unsqueeze(0)
            #rotation of 90°
            if operation == 2:
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = ris_model.transpose(0, 1)
                ris_model = ris_model.unsqueeze(0)
                ris_model = ris_model.unsqueeze(0)
            #rotation of 270°
            if operation == 3:
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = ris_model.transpose(0, 1).flip(0)
                ris_model = ris_model.unsqueeze(0)
                ris_model = ris_model.unsqueeze(0)
            # flip along the axis 0
            if operation == 4:
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = ris_model.flip(0)
                ris_model = ris_model.unsqueeze(0)
                ris_model = ris_model.unsqueeze(0)
            # flip along the axis 1
            if operation == 5:
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = torch.squeeze(ris_model, 0)
                ris_model = ris_model.flip(1)
                ris_model = ris_model.unsqueeze(0)
                ris_model = ris_model.unsqueeze(0)
            #translation to the right of 'shift' pixels
            if operation == 6:
                image = torch.zeros(ris_model.shape)
                image[:, :, shift:ris_model.shape[2], :, :] = ris_model[:, :, 0:ris_model.shape[2] - shift,
                                                              :, :]
                ris_model = image
            # downward translation of 'shift' pixels
            if operation == 7:
                image = torch.zeros(ris_model.shape)
                image[:, :, :, shift:ris_model.shape[3], :] = ris_model[:, :, :,
                                                              0:ris_model.shape[3] - shift, :]
                ris_model = image
            # translation to the left of 'shift' pixels
            if operation == 8:
                image = torch.zeros(ris_model.shape)
                image[:, :, 0:ris_model.shape[2] - shift, :, :] = ris_model[:, :, shift:ris_model.shape[2],
                                                                  :, :]
                ris_model = image
            # upward translation of 'shift' pixels
            if operation == 9:
                image = torch.zeros(ris_model.shape)
                image[:, :, :, 0:ris_model.shape[3] - shift, :] = ris_model[:, :, :,
                                                                  shift:ris_model.shape[3], :]
                ris_model = image
            #augmented dataset number one (zoom out 1)
            if operation == 10:
                ris_model, label, file_name = dset_aug.tensor_and_info(k)
            # augmented dataset number two (zoom out 2)
            if operation == 11:
                ris_model, label, file_name = dset_aug2.tensor_and_info(k)
            #permutation of the tensor in order to fit in the model
            ris_model = ris_model.permute(0, 1, 4, 2, 3)
            #calculation of the predicted label
            output, _, _ = model(ris_model.cuda(device=device))
            output = output.unsqueeze(0)
            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat((outputs, output))
            if labels is None:
                labels = label.cuda(device=device)
            else:
                labels = torch.cat((labels, label.cuda(device=device)))
            #for loss calculation
            if j >= bs:
                j = 0
                ris = loss(outputs, labels)
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(np.argmax(outputs.detach().cpu().numpy(), axis=1))
                running_samples += len(labels)
                running_loss += ris.cpu().detach().numpy()
                optimizer.zero_grad()
                ris.backward()
                optimizer.step()
                outputs = None
                labels = None

        except Exception as ex:
            print(ex)
        continue
    return true_labels, predicted_labels, running_loss, running_samples

def main(args):
    #inizializaiont of the environment and set of seed
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #pdb.set_trace()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    shuffle=True
    #if mode is TRAIN
    if args.mode == "TRAIN":
        (num_classes, train_loader, valid_loader, test_loader) = loaders(data_dir=args.data_dir,
                                                                         val_dir=args.val_dir,
                                                                         test_dir=args.test_dir,
                                                                         batch_size=1,
                                                                         random_seed=seed,
                                                                         shuffle=shuffle,
                                                                         extension=args.ext)


        #load of the model
        model = AttentionModel(num_classes=num_classes, filters_out=args.filters_out, filters_in=args.filters_in, dropout=args.dropout,
                                device=device).to(device=device)
        #train all parameters of the model
        for param in model.parameters():
            param.requires_grad = True
        #Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,  weight_decay=1e-3)
        #train and test the model
        epoch, F1 = train(model, args, device, optimizer,
                               args.num_epoch, train_loader,
                               valid_loader, test_loader, args.model_path,
                              args.model_path_fin, args.batch_size)
        print("Best model obtained at: {} with F1 = {}".format(str(epoch), F1))


    if args.mode == "TEST":
        dataset_test = TensorDataset(args.test_dir, args.ext) #test dataset
        num_classes = len(dataset_test.classes) #number of classes
        num_sample_test = len(dataset_test) #number of samples in test set
        indices_test = list(range(num_sample_test)) #indices of samples in test set
        test_sampler = SubsetRandomSampler(indices_test) #Samples elements randomly in test set

        # test dataset loader
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler)
        
        dset_test = ImageDataset(root_dir=args.test_dir, patch=args.patch_size, scale=args.patch_scale,
                             overlap=0, device=device, extension=args.ext)
        
                #load of the model
        #model = AttentionModel(num_classes=num_classes, filters_out=args.filters_out, filters_in=args.filters_in, dropout=args.dropout,
        #                        device=device).to(device=device)

        # Load the state dict
        #model.load_state_dict(torch.load('model_state_dict.pt'))
        model = torch.load(args.model_path)
        print(model)
        #pdb.set_trace()
        accuracy, precision, recall, f1, conf_mat =  test(model,test_loader,device, dset_test, args.save_attention)
        print("Test/Accuracy={}, Mean(Precision)={}, Mean(recall)={}, Mean(F1)={}, CONF={}".format(
                 accuracy, np.mean(precision),np.mean(recall),  np.mean(f1), conf_mat))



    #if mode is TENSOR
    if args.mode == "TENSOR":

        model=None
        # if the model chosen for the creation of the tensor is RESNET18
        if args.model_type == "RESNET18":
            #if the original pretained model is chosen
            if args.model_pretrained:
                model = torchvision.models.resnet18(pretrained=True)
            # if a new pretained model is chosen
            else:
                model = torchvision.models.resnet18()
                model.fc = nn.Linear(512, 3)
                model.load_state_dict(torch.load(args.model_path))
            #parameters are not trained
            for param in model.parameters():
                param.requires_grad = False
            #last layer for the classification is deleted: only features are extracted to create the tensor
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            model.cuda(device=device)
        # if the model chosen for the creation of the tensor is RESNET34
        if args.model_type == "RESNET34":
            # if the original pretained model is chosen
            if args.model_pretrained:
                model = torchvision.models.resnet34(pretrained=True)
            # if a new pretrained model is chosen
            else:
                model = torchvision.models.resnet34()
                model.fc = nn.Linear(512, 3)
                model.load_state_dict(torch.load(args.model_path))
            # parameters are not trained
            for param in model.parameters():
                param.requires_grad = False
            # last layer for the classification is deleted: only features are extracted to create the tensor
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
            model.cuda(device=device)
        #creation of the tensors
        tensors_creation(model,args,device=device)
    print("END")



if __name__ == '__main__':
    MODEL_PATH = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/models/bs2_ep50.pt"
    DATA_DIR = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/datasets/train_dataset"
    #DATA_DIR = "/gladstone/finkbeiner/steve/work/data/xdp_orig_copy"
    #EST_DIR =DATA_DIR
    TEST_DIR = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/datasets/test_dataset"
    #DATA_DIR = TEST_DIR 
    SAVE_DIR = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/datasets/tensors"
    RESULTS_DIR = "/gladstone/finkbeiner/steve/work/data/npsad_data/monika/Antibodies_detection/output"
    parser = argparse.ArgumentParser(description='Training a model')

    # General parameters
    parser.add_argument('--model_type', choices=['RESNET18','RESNET34'],default="RESNET34",help="Models used to create the Tensor_U [RESNET18,RESNET34] ")
    parser.add_argument('--model_pretrained', help='if original pretrained model this parameter should be set to True')
    parser.add_argument('--model_path', default=MODEL_PATH, help='path of the model saved for each epoch')
    parser.add_argument('--model_path_fin', default=MODEL_PATH, help='path of the final saved model')
    parser.add_argument('--data_dir',default=DATA_DIR, help='path of the train dataset')
    parser.add_argument('--val_dir', default=TEST_DIR, help='path of the validation dataset')
    parser.add_argument('--test_dir', default=TEST_DIR, help='path of the test dataset')
    parser.add_argument('--aug_dir', default=DATA_DIR, help='path of the first dataset for the augmentation')
    parser.add_argument('--aug_dir2', default=DATA_DIR, help='path of the second dataset for the augmentation')
    parser.add_argument('--save_dir', default=SAVE_DIR, help='path of the directory where tensors will be saved')
    parser.add_argument('--mode', choices=['TRAIN','TENSOR','TEST'], default="TRAIN", help="possible options: TRAIN and TENSOR")
    parser.add_argument('--seed', type=int, default=1, help='Seed value')
    parser.add_argument('--gpu_list', default="0", help='number of the GPU that will be used')
    parser.add_argument('--debug', action='store_true', help='for debug mode')
    parser.add_argument('--ext', default='pth', help='extension of the structure to load: svs/png for images (mode=TENSORS) and pth for tensors (mode=TRAIN)')

    # Training parameters
    parser.add_argument('--patch_size', type=int, default=224, help='Patch Size')
    parser.add_argument('--patch_scale', type=int, default=224, help='Patch Scale')
    parser.add_argument('--num_epoch', type=int, default=100, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
    
    # Model parameters
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--filters_out', type=int, default=64, help='number of Attention Map Filters')
    parser.add_argument('--filters_in', type=int, default=512, help='number of Input Map Filters')
    
    #GradCam/Ablation cam output
    parser.add_argument('--save_attention', type=bool, default=False, help='visualize attention maps')

    args = parser.parse_args()
    main(args)
    
#python ABNN_WSI.py --mode TRAIN --data_dir '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/train/' --val_dir /gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/val/ --test_dir /gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/test/ --aug_dir '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/train/' --aug_dir2 '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/train/' --gpu_list 0 --seed 0 --model_path '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/GigaPixel-paper/model_epochs' --model_path_fin '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/GigaPixel-paper/final' --batch_size 8 --learning_rate 0.0001 --ext pth 
#python main.py --mode TRAIN --data_dir '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/data/train_tensors/' --val_dir /gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/val/ --test_dir /gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/test/ --aug_dir '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/train/' --aug_dir2 '/gladstone/finkbeiner/steve/work/data/npsad_data/vivek/Datasets/HE-data/HE-WSI-split/train/' --gpu_list 0 --seed 0 --model_path '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/GigaPixel-paper/model_epochs' --model_path_fin '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/GigaPixel-paper/final' --ext pth --segment True --classify False --images_dir '/gladstone/finkbeiner/steve/work/data/npsad_data/monika/LBD/image_dataset' --gpu_list 0 --seed 0
