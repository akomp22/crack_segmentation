clear all


Files=dir('C:\Users\Андрей\Desktop\Skynet segmentation\initial photo for train 2\*');
%Ic=imread('C:\Users\Андрей\Desktop\Skynet segmentation\initial photo for train\24206-05-2019  09;45;25.png.png');
save_folder = 'C:\Users\Андрей\Desktop\Skynet segmentation\masks2\'
i = 233;
while i<=length(Files)
    
    folder = Files(i).folder;
    name = Files(i).name;
    folder_name = string(folder)+'\'+string(name);
    Ic=imread(char(folder_name));
    imshow(Ic);
    h1=imfreehand;
    binaryImage = h1.createMask();
    imshow(binaryImage);
    pause(1);
    
    close all
    prompt = '1 if ok, 2 to repeat: ';
    x = input(prompt)
    if x==1
        i = i+3
        save_f_n = string(save_folder)+string(name);
        imwrite(binaryImage, char(save_f_n));
    end
end