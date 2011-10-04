function [ delta_e ] = iterative_intepolation( im )
    
    ret_im = uint8(bilinear_interpolation(im));
    figure();
    imshow(ret_im);
    
    compare_im = imread('demosaickTest.tif');
    
    lab_im = RGB2Lab(compare_im);
    lab_ret_im = RGB2Lab(ret_im);
    
    delta_e = deltae(lab_im , lab_ret_im)

    ret_im = uint8(malvar_interpolation(im));
    figure();
    imshow(ret_im);
    lab_ret_im = RGB2Lab(ret_im);
    
    delta_e = deltae(lab_im , lab_ret_im)
    
    
    red_mask_kernel = [0 1; 0 0];
    green_mask_kernel = [1 0; 0 1];
    blue_mask_kernel = [0 0; 1 0];
    
    green_red_mask_kernel = [1 0; 0 0];
    green_blue_mask_kernel = [0 0; 0 1];
    
    red_mask =uint8(repmat(red_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    green_mask =uint8(repmat(green_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    blue_mask =uint8(repmat(blue_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    
    green_red_mask =uint8(repmat(green_red_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    green_blue_mask =uint8(repmat(green_blue_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    
    from_y = 1211;
    to_y = 1215;
    
    from_x = 1533;
    to_x = 1547;
    
    old_red_component = uint8(ret_im(: , : , 1));
    old_green_component = uint8(ret_im(: , : , 2));
    old_blue_component = uint8(ret_im(: , : , 3));
    
    for iteration = 1:5
        red_component = uint8(ret_im(: , : , 1));
        green_component = uint8(ret_im(: , : , 2));
        blue_component = uint8(ret_im(: , : , 3));
        
        %fprintf('Red component')
        %k =red_component;
        %k =k(from_x:to_x , from_y:to_y)
        
        %fprintf('Green component')
        %k =green_component;
        %k =k(from_x:to_x , from_y:to_y)
        
        %fprintf('Blue component')
        %k =blue_component;
        %k =k(from_x:to_x , from_y:to_y)
        
        d_r = int32(red_component) - int32(green_component);
        
        
        d_r_g_v_filter = [ 0 1 0 ; 0 0 0 ; 0 1 0] / 2;
        d_r_g_h_filter = [0 0 0; 1 0 1 ; 0 0 0] / 2;
        d_r_b_filter = [1 0 1 ; 0 0 0 ; 1 0 1] / 4; 
        
        d_r_g_v_component = imfilter(d_r , d_r_g_v_filter) .* int32(green_blue_mask);
        
        %fprintf('Green Red Component');
        %k =d_r_g_v_component(from_x:to_x , from_y:to_y)
        
        d_r_g_h_component = imfilter(d_r , d_r_g_h_filter) .* int32(green_red_mask);
        
        %fprintf('Green Blue Component')
        %k =d_r_g_h_component(from_x:to_x , from_y:to_y)
        
        d_r_b_component = imfilter(d_r , d_r_b_filter) .* int32(blue_mask);
        
        %fprintf('Blue Component')
        %k =d_r_b_component(from_x:to_x , from_y:to_y)     
        
        red_component = int32(red_component .* red_mask);
        %figure()
        %imshow(uint8(red_component));
        red_component = red_component + int32(green_component .* (green_mask + blue_mask));
        %figure()
        %imshow(uint8(red_component));
   
        
        red_component = red_component + d_r_g_v_component;
        %figure()
        %imshow(uint8(red_component));
        red_component = red_component + d_r_g_h_component;
        %figure();
        %imshow(uint8(red_component));
        red_component = red_component + d_r_b_component;
        
        red_component = uint8(red_component);
        
        %red_component = uint8(int32(red_component .* red_mask) + int32(green_component .* (green_mask + blue_mask)) + d_r_g_v_component + d_r_g_h_component + d_r_b_component);
        %k =red_component(from_x:to_x , from_y:to_y)
        
        
        d_b = int32(blue_component) - int32(green_component);
        
        %fprintf('Difference of G');
        %k =d_b(from_x:to_x , from_y:to_y)
        
        d_b_g_v_filter = [  1  ;  0  ;  1 ] / 2;
        d_b_g_h_filter = [ 1 0 1 ] / 2;
        d_b_r_filter = [1 0 1 ; 0 0 0 ; 1 0 1] / 4;
        
        d_b_g_v_component = imfilter(d_b , d_b_g_v_filter) .* int32(green_red_mask);
        %fprintf('Green blue Component')
        %k =d_b_g_v_component(from_x:to_x , from_y:to_y)
        
        d_b_g_h_component = imfilter(d_b , d_b_g_h_filter) .* int32(green_blue_mask);  
        %fprintf('Green red Component')
        %k =d_b_g_h_component(from_x:to_x , from_y:to_y)
        
        d_b_r_component = imfilter(d_b , d_b_r_filter) .* int32(red_mask);
         %fprintf('Red Component')
        %k =d_b_r_component(from_x:to_x , from_y:to_y)  
        
        
        blue_component = int32(blue_component .* blue_mask);
        %figure()
        %imshow(uint8(blue_component));
        blue_component = blue_component + int32(green_component .* (green_mask + red_mask));
        %figure()
        %imshow(uint8(blue_component));
        %fprintf('Added green component');
        %k =blue_component(from_x:to_x , from_y:to_y)
        blue_component = blue_component + d_b_g_v_component;
        %figure()
        %imshow(uint8(blue_component));
        %fprintf('Added green blue component');
        %k =blue_component(from_x:to_x , from_y:to_y)
        blue_component = blue_component + d_b_g_h_component;
        %figure();
        %imshow(uint8(blue_component));
        %imshow(uint8(blue_component((from_x-50):(to_x + 50) , (from_y-50):(to_y + 50))));
        %fprintf('Added green red component');
        %k =blue_component(from_x:to_x , from_y:to_y)
        blue_component = blue_component + d_b_r_component;
  
        %fprintf('Added red component');
        %k =blue_component(from_x:to_x , from_y:to_y);
        
        blue_component = uint8(blue_component);
        
        %blue_component = uint8(int32(blue_component .* blue_mask) + int32(green_component .* (green_mask + red_mask)) + d_b_g_v_component + d_b_g_h_component + d_b_r_component);
        %k =blue_component(from_x:to_x , from_y:to_y)

        
        d_g_r_filter = [0 1 0 ; 1 0 1 ; 0 1 0] / 4;
        d_g_b_filter = [0 1 0 ; 1 0 1 ; 0 1 0] / 4;
        
        %Recompute color difference here?
        
        d_g_r_component = imfilter(d_r , d_g_r_filter) .* int32(red_mask);
        
        d_g_b_component = imfilter(d_b , d_g_b_filter) .* int32(blue_mask);
        
        green_component = uint8(int32(green_component .* green_mask) + int32(red_component .* red_mask) - d_g_r_component + int32(blue_component .* blue_mask) - d_g_b_component);

        
        
        ret_im(: , : , 1) = red_component;
        ret_im(: , : , 2) = green_component;
        ret_im(: , : , 3) = blue_component;
        
        %figure();
        %imshow(ret_im);
        
        laplacian_filter = [0 -1 0 ; -1 4 -1 ; 0 -1 0];
        
        e_r = imfilter(d_r , laplacian_filter);
        %k =e_r(from_x:to_x , from_y:to_y)
        
        e_b = imfilter(d_b , laplacian_filter);
        %k =e_b(from_x:to_x , from_y:to_y)
        
        th = 5;
        
        delta_l = 4;
        delta_h = .05;
        
        o_r_h = find(abs(e_r) < th);
        o_r_l = find(abs(e_r) > th);
        
        o_b_h = find(abs(e_b) < th);
        o_b_l = find(abs(e_b) > th);
        
        
        
        e_r = (double(red_component - old_red_component) .^ 2) .^ .5;
        e_r = [find(e_r(o_r_h) < delta_h) ; find(e_r(o_r_l) < delta_l)];
        %k =size(e_r)
        
        e_b = (double(blue_component - old_blue_component) .^ 2) .^ .5;
        e_b= [find(e_b(o_b_h) < delta_h) ; find(e_b(o_b_l) < delta_l)];
        %k =size(e_b)
        
        
        
        
    end
    
    figure()
    imshow(ret_im);
    lab_ret_im = RGB2Lab(ret_im);
    
    delta_e = deltae(lab_im , lab_ret_im)
    
    
    
end

function [delta_e] = deltae( compare_im , im)
    difference = im - compare_im;
    
    delta = sum(sum(difference .^ 2));
    
    delta_e = sqrt(delta);
end

function [ret_im] = bilinear_interpolation( im )
    ret_im = zeros(size(im , 1) , size(im , 2) , 3);

    %Define the masks for RGB as in the standard Bayer Pattern
    %Change this if using a different pattern
    red_mask_kernel = [0 1; 0 0];
    green_mask_kernel = [1 0; 0 1];
    blue_mask_kernel = [0 0; 1 0];
    
    red_mask =uint8(repmat(red_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    green_mask =uint8(repmat(green_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    blue_mask =uint8(repmat(blue_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    
    red_component = im .* red_mask;
    green_component = im .* green_mask; 
    blue_component = im .* blue_mask;
    
    %We can just interpolate green pixels
    g_filter = [0 1 0; 1 0 1; 0 1 0] / 4;
   
    int_green_component = green_component + imfilter(green_component , g_filter);
    %figure()
    %imshow(int_green_component);
    
    ret_im( : , : , 2) = int_green_component;
    
    %To interpolate blue, first we interpolate at the red pixels
    b_r_filter = [1 0 1; 0 0 0; 1 0 1] / 4;
    b_r_component = imfilter(blue_component , b_r_filter);
    
    %Then we use those red pixels to interpolate at the green pixels
    b_g_filter = [0 1 0; 1 0 1; 0 1 0] / 4;
    b_g_component = imfilter(blue_component + b_r_component , b_g_filter);
    
    int_blue_component = blue_component + b_r_component + b_g_component;
    %figure();
    %imshow(int_blue_component);
    
    ret_im( : , : , 3) = int_blue_component;
    
    %We need to do the same basic thing with red
    r_b_filter = [1 0 1; 0 0 0; 1 0 1] / 4;
    r_b_component = imfilter(red_component , r_b_filter);
    
    %figure();
    %imshow(red_component + r_b_component);
    
    r_g_filter = [0 1 0; 1 0 1; 0 1 0] / 4;
    r_g_component = imfilter(red_component + r_b_component , r_g_filter);
    
    int_red_component = red_component + r_b_component + r_g_component;
    %figure();
    %imshow(int_red_component);
    
    ret_im( : , : , 1) = int_red_component;
    
    
    
end

function [ret_im] = malvar_interpolation( im )

    ret_im = zeros(size(im , 1) , size(im , 2) , 3);

    %Define the masks for RGB as in the standard Bayer Pattern
    %Change this if using a different pattern
    red_mask_kernel = [0 1; 0 0];
    green_mask_kernel = [1 0; 0 1];
    blue_mask_kernel = [0 0; 1 0];
    
    green_red_mask_kernel = [1 0; 0 0];
    green_blue_mask_kernel = [0 0; 0 1];
    
    %Now we need to make them the size of the image. 
    red_mask =uint8(repmat(red_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    green_mask =uint8(repmat(green_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    blue_mask =uint8(repmat(blue_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    
    green_red_mask =uint8(repmat(green_red_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    green_blue_mask =uint8(repmat(green_blue_mask_kernel , size(im , 1) / 2, size(im , 2) / 2));
    
    
    from = 32;
    to = 48;
    
    %Extrace the red, green, and blue components of the original image into
    %their own matrix
    red_component = im .* red_mask;
    green_component = im .* green_mask; 
    blue_component = im .* blue_mask;
    
    %Define the intepolation matrices which we will use
    
    %Green value at red pixel location - green component (We add red in
    %later
    g_r_g_mask =[0 2 0; 2 0 2; 0 2 0] / 8;
    %Green value at blue pixel location - green component
    g_b_g_mask =[0 2 0; 2 0 2; 0 2 0] / 8;
    
    %The values interpolated from the green pixels at a red location we
    %multiply by the red mask kernel to get only the red location values
    %fprintf('Green at red pixel green component');
    g_r_g_component = imfilter(green_component , g_r_g_mask , 'replicate') .* red_mask;
    %k =g_r_g_component(from:to , from:to)

    %fprintf('Green at blue pixel green component');
    g_b_g_component = imfilter(green_component , g_b_g_mask , 'replicate') .* blue_mask;
    %k =g_b_g_component(from:to , from:to)
    
    %Green value at red pixel location - red component
    g_r_r_mask =[0 0 -1 0 0; 0 0 0 0 0; -1 0 4 0 -1; 0 0 0 0 0; 0 0 -1 0 0] / 8
    %Green value at blue pixel location - blue component
    g_b_b_mask =[0 0 -1 0 0; 0 0 0 0 0; -1 0 4 0 -1; 0 0 0 0 0; 0 0 -1 0 0] / 8
    
    %fprintf('Green at red pixel red component');
    g_r_r_component = imfilter(red_component , g_r_r_mask , 'replicate');
    %k =g_r_r_component(from:to , from:to)
    
    %fprintf('Green at blue pixel blue component');
    g_b_b_component = imfilter(blue_component , g_b_b_mask , 'replicate');
    %k =g_b_b_component(from:to , from:to)
    
    %fprintf('Green component')
    %k =green_component(from:to , from:to)
    
    %k =(g_r_g_component + g_r_r_component);
    
    %fprintf('Green at red')
    %k =k(from:to , from:to)
    
    int_green_component = green_component + g_r_g_component + g_r_r_component + g_b_g_component + g_b_b_component;
    %k =int_green_component(from:to , from:to)
    
    %figure()
    %imshow(int_green_component);
    
    
    ret_im(: , : , 2) = int_green_component;
    
    %Red at green pixel, red row blue column, green component
    r_g_r_b_g_mask =[0  0 .5 0 0; 0 -1 0 -1 0; -1 0 5 0 -1; 0 -1 0 -1 0; 0 0 .5 0 0] / 8;
    r_g_r_b_r_mask =[0 0 0; 4 0 4; 0 0 0] / 8;
    %Red at green pixel, blue row red column, green component
    r_g_b_r_g_mask =[0 0 .5 0 0; 0 -1 0 -1 0; -1 0 5 0 -1; 0 -1 0 -1 0; 0 0 .5 0 0] / 8;
    r_g_b_r_r_mask =[0 4 0; 0 0 0; 0 4 0] / 8;
    %Red at blue pixel, blue row blue column, blue component
    r_b_b_b_b_mask =[0 0 -1.5 0 0; 0 0 0 0 0; -1.5 0 6 0 -1.5; 0 0 0 0 0; 0 0 -1.5 0 0] / 8;
    r_b_b_b_r_mask =[2 0 2; 0 0 0; 2 0 2] / 8;
    
    %fprintf('Red at green pixel , red row blue column');
    r_g_r_b_g_component = imfilter(green_component , r_g_r_b_g_mask , 'replicate') .* green_red_mask;
    %k =r_g_r_b_g_component(from:to , from:to)
    r_g_r_b_r_component = imfilter(red_component , r_g_r_b_r_mask , 'replicate');
    %k =r_g_r_b_r_component(from:to , from:to)
    r_g_r_b_component = r_g_r_b_g_component + r_g_r_b_r_component;
    %k =r_g_r_b_component(from:to , from:to)
    
    %fprintf('Red at green pixel , blue row red column');
    r_g_b_r_g_component = imfilter(green_component , r_g_b_r_g_mask , 'replicate').* green_blue_mask;
    %k =r_g_b_r_g_component(from:to , from:to)
    r_g_b_r_r_component = imfilter(red_component , r_g_b_r_r_mask , 'replicate');
    %k =r_g_b_r_r_component(from:to , from:to)
    r_g_b_r_component = r_g_b_r_g_component + r_g_b_r_r_component;
    %k =r_g_b_r_component(from:to , from:to)
    
    %fprintf('Red at blue pixel');
    r_b_b_b_b_component = imfilter(blue_component , r_b_b_b_b_mask , 'replicate') .* blue_mask;
    %k =r_b_b_b_b_component(from:to , from:to)
    r_b_b_b_r_component = imfilter(red_component , r_b_b_b_r_mask , 'replicate');
    %k =r_b_b_b_r_component(from:to , from:to)
    r_b_b_b_component = r_b_b_b_b_component + r_b_b_b_r_component;
    %k =r_b_b_b_component(from:to , from:to)
    
    int_red_component = red_component + r_g_r_b_component + r_g_b_r_component + r_b_b_b_component;
    %fprintf('Red interpolation');
    %k =int_red_component(from:to , from:to)
    
    %figure();
    %imshow(int_red_component);
    
    
    
    ret_im(: , : , 1) = int_red_component;
    
    %Blue at green pixel, blue row red column, green component
    b_g_b_r_g_mask =[0  0 .5 0 0; 0 -1 0 -1 0; -1 0 5 0 -1; 0 -1 0 -1 0; 0 0 .5 0 0] / 8;
    b_g_b_r_b_mask =[0 0 0; 4 0 4; 0 0 0] / 8;
    %Blue at green pixel, red row blue column, green component
    b_g_r_b_g_mask =[0 0 .5 0 0; 0 -1 0 -1 0; -1 0 5 0 -1; 0 -1 0 -1 0; 0 0 .5 0 0] / 8;
    b_g_r_b_b_mask =[0 4 0; 0 0 0; 0 4 0] / 8;
    %Blue at red pixel, red row red column, red component
    b_r_r_r_r_mask =[0 0 -1.5 0 0; 0 0 0 0 0; -1.5 0 6 0 -1.5; 0 0 0 0 0; 0 0 -1.5 0 0] / 8;
    b_r_r_r_b_mask =[2 0 2; 0 0 0; 2 0 2] / 8;
    
    %fprintf('Blue at green pixel , blue row red column');
    b_g_b_r_g_component = imfilter(green_component , b_g_b_r_g_mask , 'replicate') .* green_blue_mask;
    %k =b_g_b_r_g_component(from:to , from:to)
    b_g_b_r_b_component = imfilter(blue_component , b_g_b_r_b_mask , 'replicate');
    %k =b_g_b_r_b_component(from:to , from:to)
    b_g_b_r_component = b_g_b_r_g_component + b_g_b_r_b_component;
    %k =b_g_b_r_component(from:to , from:to)
    
    %fprintf('Blue at green pixel , red row blue column');
    b_g_r_b_g_component = imfilter(green_component , b_g_r_b_g_mask , 'replicate') .* green_red_mask;
    %k =b_g_r_b_g_component(from:to , from:to)
    b_g_r_b_b_component = imfilter(blue_component , b_g_r_b_b_mask , 'replicate');
    %k =b_g_r_b_b_component(from:to , from:to)
    b_g_r_b_component = b_g_r_b_g_component + b_g_r_b_b_component;
    %k =b_g_r_b_component(from:to , from:to)
    
    %fprintf('Blue at red pixel');
    b_r_r_r_r_component = imfilter(red_component , b_r_r_r_r_mask , 'replicate') .* red_mask;
    %k =b_r_r_r_r_component(from:to , from:to)
    b_r_r_r_b_component = imfilter(blue_component , b_r_r_r_b_mask , 'replicate');
    %k =b_r_r_r_b_component(from:to , from:to)
    b_r_r_r_component = b_r_r_r_r_component + b_r_r_r_b_component;
    %k =b_r_r_r_component(from:to , from:to)
    
    int_blue_component = blue_component + b_g_b_r_component + b_g_r_b_component + b_r_r_r_component;
    %fprintf('Blue interpolation');
    %k =int_blue_component(from:to , from:to)
    
    %figure();
    %imshow(int_blue_component);
    
    ret_im(: , : , 3) = int_blue_component;
end
