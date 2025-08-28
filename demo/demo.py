from tkinter import messagebox, filedialog as fd
import customtkinter
from PIL import Image
import os

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms

# ----------------- Modelli -----------------
def get_resnet_model():
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_classes = 11
    model.fc = nn.Linear(512, num_classes)
    model.num_classes = num_classes
    return model

def get_squeezenet_model():
    model = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
    model.classifier[1] = nn.Conv2d(512, 11, kernel_size=(1,1), stride=(1,1))
    model.num_classes = 11
    return model

def get_efficientnet_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(1280, 11)
    return model

class_names = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit']

# ----------------- Trasformazioni -----------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

# ----------------- GUI -----------------
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.title("Demo FoodVision")
img_path = ''
photo_label = None
selected_model = customtkinter.StringVar(value="ResNet18")

# ----- Funzioni -----
def openfilename():
    filename = fd.askopenfilename(title='Scegli immagine', filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return filename

def caricaImg():
    global photo_label, img_path
    img_path = openfilename()
    if img_path == '':
        return
    if not os.path.isfile(img_path):
        messagebox.showerror("Errore", "Seleziona un file immagine valido!")
        return

    img_pil = Image.open(img_path)
    # Ridimensiona se troppo grande
    max_size = 400
    if img_pil.width > max_size or img_pil.height > max_size:
        if img_pil.width > img_pil.height:
            width = max_size
            height = int((width * img_pil.height) / img_pil.width)
        else:
            height = max_size
            width = int((height * img_pil.width) / img_pil.height)
        resized_img = img_pil.resize((width, height), Image.LANCZOS)
    else:
        resized_img = img_pil.copy()

    ctk_img = customtkinter.CTkImage(resized_img, size=resized_img.size)

    # Se la label non esiste, la creo, altrimenti aggiorno l’immagine
    if photo_label is None:
        global frame1
        photo_label = customtkinter.CTkLabel(frame1, image=ctk_img, text="")
        photo_label.image = ctk_img
        photo_label.place(relx=0.5, rely=0.5, anchor='center')
    else:
        photo_label.configure(image=ctk_img)
        photo_label.image = ctk_img

def classify():
    global img_path
    if img_path == '' or not os.path.isfile(img_path):
        messagebox.showerror("Errore", "Prima seleziona un'immagine valida!")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_choice = selected_model.get()


    # Carica modello corretto
    if model_choice == "ResNet18":
        model = get_resnet_model()
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_file = os.path.join(BASE_DIR, "modelli", "resnet_finetuning_lr=0.0001", "resnet_finetuning_lr=0.0001-30.pth")
    elif model_choice == "SqueezeNet":
        model = get_squeezenet_model()
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_file = os.path.join(BASE_DIR, "modelli", "squeezenet_finetuning_lr=0.0001", "squeezenet_finetuning_lr=0.0001-30.pth")
    elif model_choice == "EfficientNet-B0":
        model = get_efficientnet_model()
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_file = os.path.join(BASE_DIR, "modelli", "efficientnet_finetuning_lr=0.0001", "efficientnet_finetuning_lr=0.0001-30.pth")

    else:
        messagebox.showerror("Errore", "Seleziona un modello valido!")
        return

    # Controllo che il file esista
    if not os.path.isfile(model_file):
        messagebox.showerror("Errore", f"File modello non trovato:\n{model_file}")
        return

    # Carica modello
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    model.to(device)

    # Carica immagine
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform_test(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    label1.configure(text=f"Classe predetta: {class_names[predicted_class]}")

def on_closing2():
    if messagebox.askokcancel("Esci", "Sei sicuro di voler uscire?"):
        root.destroy()
        exit()

# ----- GUI Layout -----
root.protocol("WM_DELETE_WINDOW", on_closing2)
root.geometry("860x550")
w2, h2 = 860, 550
ws, hs = root.winfo_screenwidth(), root.winfo_screenheight()
x, y = (ws/2) - (w2/2), (hs/2) - (h2/2)
root.geometry('%dx%d+%d+%d' % (w2, h2, x, y))
# ----- Layout griglia -----
root.columnconfigure(0, weight=3)   # colonna sinistra grande
root.columnconfigure(1, weight=1)   # colonna destra divisa
root.rowconfigure(0, weight=1)
root.rowconfigure(1, weight=1)


# Frame immagine
frame1 = customtkinter.CTkFrame(root)
frame1.grid(column=0, row=0, rowspan=2, sticky='nsew', padx=20, pady=20)

# Frame comandi
frameComandi1 = customtkinter.CTkFrame(root)
frameComandi1.grid(column=1, row=0, sticky="nsew", padx=20, pady=(20,10))

frameComandi2 = customtkinter.CTkFrame(root)
frameComandi2.grid(column=1, row=1, sticky="nsew", padx=20, pady=(10,20))

# Widget frame1
button1 = customtkinter.CTkButton(master=frame1, text="Aggiungi immagine", command=caricaImg)
button1.place(relx=0.5, rely=0.5, anchor='center')

# Widget frameComandi1
label = customtkinter.CTkLabel(master=frameComandi1, text="Pannello dei comandi", font=('Arial', 20))
label.pack(pady=(10,20))

frame_model = customtkinter.CTkFrame(master=frameComandi1, fg_color="transparent")
frame_model.pack(pady=10)
label_model = customtkinter.CTkLabel(frame_model, text="Scegli modello:")
label_model.grid(row=0, column=0, padx=(0,10))

model_menu = customtkinter.CTkOptionMenu(
    master=frame_model,
    values=["SqueezeNet", "ResNet18", "EfficientNet-B0"],
    variable=selected_model
)
model_menu.grid(row=0, column=1)

comparison = customtkinter.CTkButton(master=frameComandi1, text="Cambia immagine", command=caricaImg)
comparison.pack(pady=20)

classificaImmagine = customtkinter.CTkButton(master=frameComandi1, text="Classifica Immagine", command=classify)
classificaImmagine.pack(pady=10)

# Widget frameComandi2
label1 = customtkinter.CTkLabel(master=frameComandi2, text="Risultato classificazione", font=('Arial', 18))
label1.pack(pady=(50,20))

root.mainloop()
