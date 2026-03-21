library(igraph)
library(Matrix)
require(graphics)
library(svglite)


#' @title Obtain adjacency matrices for \code{igraph} plotting
#' @description Takes the alpha active path matrices for each layer of the SICNN and converts
#' them to adjacency matrices so that they can be plotted with igraph.
#' @param model An instance of \code{SICNN_Net} with input_skip enabled. 
#' @return A list of adjacency matrices, one for each hidden layer and the output layer. 
#' @keywords internal
get_adj_mats <- function(model){
  mats_out <-list()
  i <- 1
  for(l in model$layers$children){
    alp <- t(as.matrix(l$alpha_active_path))
    adj_mat <- matrix(0,nrow = sum(dim(alp)),ncol = sum(dim(alp))) #initialize empty matrix
    first_dim <- seq_len(dim(alp)[1])
    second_dim <- (dim(alp)[1] +1):sum(dim(alp))
    adj_mat[first_dim,second_dim] <- alp
    mats_out[[i]] <- adj_mat
    i <- i + 1
  } #do the same for the output layer
  alp_out <- t(as.matrix(model$out_layer$alpha_active_path))
  adj_mat_out <- matrix(0,nrow = sum(dim(alp_out)),ncol = sum(dim(alp_out))) #
  first_dim <- seq_len(dim(alp_out)[1])
  second_dim <- (dim(alp_out)[1] +1):sum(dim(alp_out))
  adj_mat_out[first_dim,second_dim] <- alp_out
  mats_out[[i]] <- adj_mat_out
  
  
  return(mats_out)
}

#' @title Function for plotting nodes in the network between two layers. 
#' @description Takes care of the three possible cases. Both layers have even
#' number of neurons, both layers have odd numbers, or one of each. 
#' @param N integer, number of neurons in the first layer.
#' @param N_u integer, number of neurons in the second layer.
#' @param input_positions Positions of the neurons in the input layer.
#' @param neuron_spacing How much space between the neurons. 
#' @return Positions of the second layer. 
#' @keywords internal
assign_within_layer_pos<- function(N,N_u,input_positions,neuron_spacing){
  if(N %% 2 == 0 && N_u %% 2 == 0){ #if both layers have even number of neurons
    N_u_center <- stats::median(input_positions)
    N_u_start_pos <- N_u_center + neuron_spacing / 2 - (N_u /2 * neuron_spacing) #add the half space, then subtract half of the array to get to start point
    N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)

  } 
  
  if(N %% 2 != 0 && N_u %% 2 != 0){ #if both layers have odd number of neurons
    N_u_center <- stats::median(input_positions)
    N_u_start_pos <- N_u_center - ((N_u - 1) / 2) * neuron_spacing #just need to figure out how many neurons to the left of the median one
    N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)
  } 
  
  if((N + N_u) %% 2 != 0){ #in the case of even and odd number of neurons. Even + odd = odd
    if(N > N_u){ #in this case, N_u is odd
      N_u_center <- stats::median(input_positions)
      N_u_start_pos <- N_u_center + neuron_spacing / 2 - (N_u /2 * neuron_spacing) 
      N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)
    }
    if(N < N_u){ #in this case, N_u is even
      N_u_center <- stats::median(input_positions)
      N_u_start_pos <- N_u_center - ((N_u - 1) / 2) * neuron_spacing 
      N_u_positions <- seq(from = N_u_start_pos, length.out = N_u,by = neuron_spacing)
    }
  }
    return(N_u_positions)
}


#' @title Assign names to nodes.
#' @description Internal helper function to assign descriptive names to nodes used for plotting.
#' @param model A trained \code{SICNN_Net} model with input-skip.
#' @return A list of adjacency matrices with the correct names.
#' @keywords internal
assign_names<- function(model){#assign names to the nodes before plotting
  alphas <- get_adj_mats(model)
  sizes <- model$sizes
  for(i in seq_along(alphas)){
    mat_names <- c()
    if(i == 1){ #for the input layer
      for(j in 1:sizes[1]){ #first the x_i
        name <- paste('x',j-1,'_',i-1,sep = '') #i-1 because x belongs to the first (input layer)
        mat_names <- c(mat_names,name)
      }
      for(j in 1:sizes[i + 1]){#then the u
        name <- paste('u',j-1,'_',i,sep = '')
        mat_names <- c(mat_names,name)
      }
      
      colnames(alphas[[i]]) <- mat_names
    }
    else if(i < length(alphas)){#all other layers except the last
      
      mat_names <- c()
      for(j in 1:sizes[i]){#N - n_input is the number of neurons in the hidden layer
        name <- paste('u',j-1,'_',i-1,sep = '')
        mat_names <- c(mat_names,name)
      }
      for(j in 1:sizes[1]){#the input skip x
        name <- paste('x',j-1,'_',i-1,sep = '')
        mat_names <- c(mat_names,name)
      }
      for(j in 1:sizes[i + 1]){#the hidden neurons for the next layer
        name <- paste('u',j-1,'_',i,sep = '')
        mat_names <- c(mat_names,name)
      }
      colnames(alphas[[i]]) <- mat_names
      
    }
    else{#the last layer note: this is almost the same as above, could join them together??
      mat_names <- c()
      for(j in 1:sizes[i]){#N - n_input is the number of neurons in the hidden layer
        name <- paste('u',j-1,'_',i-1,sep = '')
        mat_names <- c(mat_names,name)
      }
      for(j in 1:sizes[1]){#the input skip x
        name <- paste('x',j-1,'_',i-1,sep = '')
        mat_names <- c(mat_names,name)
      }
      for(j in 1:sizes[i + 1]){#the hidden neurons for the next layer
        name <- paste('y',j-1,sep = '')
        mat_names <- c(mat_names,name)
      }
      colnames(alphas[[i]]) <- mat_names
      
    }
    
  }
  return(alphas)
  
}



#' @title Function to plot an input skip structure after removing weights in non-active paths.
#' @description Uses igraph to plot.
#' @param model A trained \code{SICNN_Net} model with input_skip enabled. 
#' @param layer_spacing numeric, spacing in between layers.
#' @param neuron_spacing numeric, spacing between neurons within a layer.
#' @param vertex_size numeric, size of the neurons. 
#' @param label_size numeric, size of the text within neurons. 
#' @param edge_width numeric, width of the edges connecting neurons.
#' @param save_svg the path where the plot will be saved if save_svg is not null.
#' @examples
#' \donttest{
#' sizes <- c(2,3,3,1)
#' problem <- 'regression'
#' device <- 'cpu'
#' torch::torch_manual_seed(0)
#' model <- SICNN_Net(problem_type = problem, sizes = sizes,
#'                    input_skip = TRUE, device = device)
#' x <- torch::torch_randn(3,2)
#' b <- torch::torch_rand(2)
#' y <- torch::torch_matmul(x,b)$unsqueeze(2)
#' train_data <- torch::tensor_dataset(x,y)
#' train_loader <- torch::dataloader(train_data, batch_size = 3)
#' train_SICNN(epochs = 1, SICNN = model, lr = 0.01, train_dl = train_loader, n_train = 3)
#' SICNN_plot(model, 1, 1, 14, 1)
#' }
#' @return This function produces plots as a side effect and does not return a value.
#' @export
SICNN_plot <- function(model,layer_spacing = 1,neuron_spacing = 1,vertex_size = 10,label_size = 0.5,edge_width = 0.5,save_svg = NULL){
  
  if (!is.null(save_svg)) {
    # Open SVG device
    svglite::svglite(save_svg, width = 5, height = 4)
  }
  
  graph <- assign_names(model) #the graph with names neurons, given some model with alpha matrices
  g <- igraph::make_empty_graph(n = 0) #initialize empty graph
  for(L in seq_along(graph)){
    g <- g +  igraph::graph_from_adjacency_matrix(graph[[L]],mode = 'directed')
  }
  plot_points <- matrix(0,nrow = length(g),ncol = 2) #x,y coordinates for all neurons in g
  layer_positions <- seq(from = 0,by = - layer_spacing,length.out = length(model$sizes)) #position for each layer
  index_start <- 0 
  dim_1_pos <- 0
  i <- 1
  for(s in model$sizes){
    
    if(i == 1){
      plot_points[1:model$sizes[i],2] <- layer_positions[i] #input layer coords
      index_start <- model$sizes[i] + 1 #where to start next layer
      dim_1_pos <- seq(from = 0,length.out = model$sizes[i],by = neuron_spacing)#coords within input layer
      plot_points[1:model$sizes[i],1] <- dim_1_pos 
      
    }
    else if(i < length(model$sizes)){#all other layers except the last
      
      plot_points[(index_start:(index_start + model$sizes[1] + model$sizes[i]-1)),2] <- layer_positions[i]
      #N = size of prev layer #N_u size of current layer
      dim_1_pos <- assign_within_layer_pos(N = length(dim_1_pos),N_u = model$sizes[1] + model$sizes[i],
                                           input_positions = dim_1_pos,neuron_spacing = neuron_spacing)
      
      
      
      plot_points[(index_start:(index_start + model$sizes[1] + model$sizes[i]-1)),1] <- dim_1_pos
      index_start <- index_start + model$sizes[1] + model$sizes[i] 
      
      
      
    }
    else{ #output layer
      dim_1_pos <- assign_within_layer_pos(N = length(dim_1_pos),N_u = model$sizes[length(model$sizes)],
                                           input_positions = dim_1_pos,neuron_spacing = neuron_spacing)
      plot_points[(index_start:(dim(plot_points)[1])),1] <- dim_1_pos
      plot_points[(index_start:(dim(plot_points)[1])),2] <- layer_positions[i]
      
      
    }
    i <- i + 1
    
  }
  #assign colors based on what type of neuron it is
  for(z in seq_along(igraph::V(g))){ 
    string <- igraph::V(g)[z]
    if(grepl('x',string$name)){ #for input neurons
      igraph::V(g)[z]$color <- '#D5E8D4'
      igraph::V(g)[z]$frame.color <- '#D5E8D4'#change color of boundary too
    }
    else if(grepl('u',string$name)){ #hidden neurons
      igraph::V(g)[z]$color <- '#ADD8E6'
      igraph::V(g)[z]$frame.color <- '#ADD8E6'
    }
    else{
      igraph::V(g)[z]$color <- '#F8CECC' #output neurons
      igraph::V(g)[z]$frame.color <- '#F8CECC'
    }
    
    
  }
  
  plot(g,vertex.size = vertex_size,vertex.label.cex = label_size, 
       edge.color = 'black',vertex.label.color='black',
       edge.width = edge_width, layout = -plot_points[,2:1],edge.arrow.mode = '-',margin = 0.0,asp = 0)
 
  if(!is.null(save_svg)) {
    message(paste("Plot saved as", save_svg))
    on.exit(grDevices::dev.off()) # ensures the device closes even if an error occurs
  }
  
}


#' @title Plot SICNN Convolutional Network
#' @description Plots the Convolutional Network topology using a single square node abstraction 
#' for each convolutional layer, expanding into a sparse bipartite visualization for the fully connected parts.
#' @param x An object of class \code{SICNN_ConvNet}
#' @param threshold Numeric threshold for identifying active edges (default 0.5).
#' @param ... Additional arguments.
#' @return Invisible NULL.
#' @method plot SICNN_ConvNet
#' @export
plot.SICNN_ConvNet <- function(x, threshold=0.5, ...) {
  if (!requireNamespace("igraph", quietly = TRUE)) {
    message("The igraph package is required for network plotting.")
    return(invisible(x))
  }
  
  # Extract Dimensions
  n_in <- x$fc1$in_features
  n_hid <- x$fc1$out_features
  n_out <- x$fc2$out_features
  
  # Construct Graph Nodes: Input(1) -> Conv1(1) -> Conv2(1) -> FC_in(n_in) -> FC_hid(n_hid) -> FC_out(n_out)
  total_nodes <- 3 + n_in + n_hid + n_out
  adj_mat <- matrix(0, nrow=total_nodes, ncol=total_nodes)
  
  # Map offsets
  o_in <- 1
  o_c1 <- 2
  o_c2 <- 3
  o_fc_in <- 4
  o_fc_hid <- o_fc_in + n_in
  o_fc_out <- o_fc_hid + n_hid
  
  # Abstract edges
  adj_mat[o_in, o_c1] <- 1
  adj_mat[o_c1, o_c2] <- 1
  adj_mat[o_c2, o_fc_in:(o_fc_in+n_in-1)] <- 1
  
  # Sparse FC1 Alpha
  alp_fc1 <- t(as.matrix(x$fc1$alpha_active_path$detach()$cpu() > threshold)) * 1
  adj_mat[o_fc_in:(o_fc_in+n_in-1), o_fc_hid:(o_fc_hid+n_hid-1)] <- alp_fc1
  
  # Sparse FC2 Alpha
  alp_fc2 <- t(as.matrix(x$fc2$alpha_active_path$detach()$cpu() > threshold)) * 1
  adj_mat[o_fc_hid:(o_fc_hid+n_hid-1), o_fc_out:(o_fc_out+n_out-1)] <- alp_fc2
  
  g <- igraph::graph_from_adjacency_matrix(adj_mat, mode="directed")
  
  # Layout Coordinates
  plot_points <- matrix(0, nrow=total_nodes, ncol=2)
  plot_points[o_in, 1] <- 1
  plot_points[o_c1, 1] <- 2
  plot_points[o_c2, 1] <- 3
  plot_points[o_fc_in:(o_fc_in+n_in-1), 1] <- 4
  plot_points[o_fc_hid:(o_fc_hid+n_hid-1), 1] <- 5
  plot_points[o_fc_out:(o_fc_out+n_out-1), 1] <- 6
  
  get_y <- function(n) { if(n==1) 0 else seq(-n/2, n/2, length.out=n) }
  plot_points[o_in, 2] <- 0
  plot_points[o_c1, 2] <- 0
  plot_points[o_c2, 2] <- 0
  plot_points[o_fc_in:(o_fc_in+n_in-1), 2] <- get_y(n_in)
  plot_points[o_fc_hid:(o_fc_hid+n_hid-1), 2] <- get_y(n_hid)
  plot_points[o_fc_out:(o_fc_out+n_out-1), 2] <- get_y(n_out)
  
  # Aestetics
  v_colors <- c('#D5E8D4', '#D5E8D4', '#ADD8E6', 
                rep('#ADD8E6', n_in), rep('#ADD8E6', n_hid), rep('#F8CECC', n_out))
  
  v_shapes <- c("square", "square", "square", 
                rep("circle", n_in), rep("circle", n_hid), rep("circle", n_out))
  
  sz_flat <- max(0.5, 15 / sqrt(n_in))
  sz_hid <- max(1, 15 / sqrt(n_hid))
  sz_out <- max(3, 15 / sqrt(n_out))
  v_sizes <- c(15, 15, 15, rep(sz_flat, n_in), rep(sz_hid, n_hid), rep(sz_out, n_out))
  
  oldpar <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(oldpar))
  graphics::par(mar=c(0,0,2,0))
  
  igraph::plot.igraph(g, layout=plot_points, vertex.size=v_sizes, vertex.label=NA, 
       vertex.shape=v_shapes, edge.arrow.size=0.1, edge.width=0.2, 
       vertex.color=v_colors, vertex.frame.color="grey30", asp = 0,
       main="SICNN Convolutional Sparse Topology")
  
  invisible(x)
}

