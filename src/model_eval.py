import os
from PIL import Image
import torchvision.transforms as transforms

"""# Query the Pinecone index and make a prediction"""

transform = transforms.Compose([
    transforms.Resize((224,224)),  # Resize all images to the size of the finetuning transformation
    transforms.ToTensor(),  # Transform the image data into tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])

# Remove the last layer (classifier) to get the feature extractor
feature_extractor = nn.Sequential(*list(model.children())[:-1])
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

# Define a function to preprocess and get the vector representation
def get_vector_representation(image_path):
    with Image.open(image_path) as img:
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        feature_extractor.eval()
        with torch.no_grad():
            batch_t = batch_t.to(device)
            outputs = feature_extractor(batch_t)
            embeddings = outputs.view(outputs.size(0), -1).cpu().numpy()
            return embeddings[0]

def get_image_path(predicted_class):
    # List of directories to search for the image
    base_dirs = ["/content/drive/MyDrive/ground_truth_images_for_CNN_pkmn",
                 "/content/drive/MyDrive/pkmn_cards_jpn"]

    # Iterate through each directory in the list
    for base_dir in base_dirs:
        # Iterate through all the filenames in the current directory
        for filename in os.listdir(base_dir):
            # Check if the first element of the filename matches the class name
            # and ensure the filename does not start with "aug_"
            if filename.split('_')[0] == predicted_class and not filename.startswith("aug_"):
                return os.path.join(base_dir, filename)

# Initialize predictions dict
predictions = {}

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

"""# Combined Testing Script for all three directories"""

# List of test directories
test_dirs = [
    "/content/drive/MyDrive/visual_search_test_set/visual_search_test_set_normal",
    "/content/drive/MyDrive/visual_search_test_set/visual_search_test_set_holo",
    "/content/drive/MyDrive/visual_search_test_set/visual_search_test_set_full_art",
]

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

total_accurate_predictions = 0
total_total_predictions = 0

for test_dir in test_dirs:
    accurate_predictions = 0
    total_predictions = 0

    for item_name in os.listdir(test_dir):
        item_path = os.path.join(test_dir, item_name)
        if os.path.isfile(item_path) and item_path.lower().endswith(('.png', '.jpg')):
            total_predictions += 1
            vector_representation = get_vector_representation(item_path)

            # Query Pinecone index for top 2 matches
            query_results = index.query(vector=[vector_representation.tolist()], namespace='pkmn-eng', top_k=1)
            actual_class = item_name.split('_')[0]
            match_found = False

            for match in query_results.matches:
                predicted_class = match.id.split('_')[0]
                print(f"Predicted class for {item_name}: {predicted_class}")

                if predicted_class == actual_class:
                    accurate_predictions += 1
                    match_found = True
                    break  # Break the loop if a match is found in the top 2

            if match_found:
                # Load and display images only if there's a match
                actual_img = mpimg.imread(item_path)
                predicted_path = get_image_path(predicted_class)

                if predicted_path and os.path.exists(predicted_path):
                    try:
                        predicted_img = mpimg.imread(predicted_path)
                        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                        ax[0].imshow(actual_img)
                        ax[0].set_title("Actual Image")
                        ax[1].imshow(predicted_img)
                        ax[1].set_title("Predicted Image")
                        plt.show()
                    except SyntaxError as e:
                        print(f"Error reading predicted image {predicted_path}: {e}")
                else:
                    print(f"No matching image found for predicted class: {predicted_class}")
        else:
            print(f"Skipping non-image file: {item_name}")

    accuracy = accurate_predictions / total_predictions if total_predictions else 0
    print(f"Directory: {test_dir}")
    print(f"Accurate Predictions: {accurate_predictions}, Total Predictions: {total_predictions}")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    total_accurate_predictions += accurate_predictions
    total_total_predictions += total_predictions

total_accuracy = total_accurate_predictions / total_total_predictions if total_total_predictions else 0
print(f"Total Accurate Predictions: {total_accurate_predictions}")
print(f"Total Predictions: {total_total_predictions}")
print(f"Total Combined Accuracy: {total_accuracy * 100:.2f}%")

"""# Prediction at K = 2"""

for test_dir in test_dirs:
    accurate_predictions = 0
    total_predictions = 0

    for item_name in os.listdir(test_dir):
        item_path = os.path.join(test_dir, item_name)
        if os.path.isfile(item_path) and item_path.lower().endswith(('.png', '.jpg', 'jpeg')):
            total_predictions += 1
            vector_representation = get_vector_representation(item_path)

            # Query Pinecone index for top 2 matches
            query_results = index.query(vector=[vector_representation.tolist()], top_k=2)
            actual_class = item_name.split('_')[0]
            matched = False  # Flag to track if any of the top 2 predictions match the actual class

            for match in query_results.matches:
                predicted_class = match.id.split('_')[0]
                print(f"Predicted class for {item_name}: {predicted_class}")

                # Check if prediction is accurate
                if predicted_class == actual_class:
                    accurate_predictions += 1
                    matched = True
                    break  # Exit loop early if a match is found

            # Only proceed to display images if a match was found
            if matched:
                # Displaying the actual vs. predicted images for the first accurate prediction
                actual_img = mpimg.imread(item_path)
                predicted_path = get_image_path(predicted_class)  # Assuming this function exists

                if predicted_path:
                    predicted_img = mpimg.imread(predicted_path)
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(actual_img)
                    ax[0].set_title("Actual Image")
                    ax[1].imshow(predicted_img)
                    ax[1].set_title("Predicted Image")
                    plt.show()
                else:
                    print(f"No matching image found for predicted class: {predicted_class}")

    accuracy = accurate_predictions / total_predictions if total_predictions != 0 else 0
    print(f"Directory: {test_dir}")
    print(f"Accurate Predictions: {accurate_predictions}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    # Add to total counts
    total_accurate_predictions += accurate_predictions
    total_total_predictions += total_predictions

# Calculate and print total combined accuracy
total_accuracy = total_accurate_predictions / total_total_predictions if total_total_predictions != 0 else 0
print(f"Total Accurate Predictions: {total_accurate_predictions}")
print(f"Total Predictions: {total_total_predictions}")
print(f"Total Combined Accuracy: {total_accuracy * 100:.2f}%")
pass

"""# Testing for Normal"""

# Initialize predictions dict and other variables for accuracy calculation
test_dir = "/content/drive/MyDrive/visual_search_test_set/visual_search_test_set_normal"
accurate_predictions = 0
total_predictions = 0

for item_name in os.listdir(test_dir):
    item_path = os.path.join(test_dir, item_name)
    if os.path.isfile(item_path):
        total_predictions += 1
        vector_representation = get_vector_representation(item_path)

        # Query Pinecone index
        query_results = index.query(vector=[vector_representation.tolist()], top_k=1)
        if query_results.matches:
            predicted_class = query_results.matches[0].id.split('_')[0]
            actual_class = item_name.split('_')[0]
            print(f"Predicted class for {item_name}: {predicted_class}")

        # Increment correct prediction count if prediction is accurate
            if predicted_class == actual_class:
                accurate_predictions += 1

        # Displaying the actual vs. predicted images for all predictions
        actual_img = mpimg.imread(item_path)
        predicted_path = get_image_path(predicted_class)

        if predicted_path:
            predicted_img = mpimg.imread(predicted_path)
        else:
            print(f"No matching image found for predicted class: {predicted_class}")
            continue

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(actual_img)
        ax[0].set_title("Actual Image")
        ax[1].imshow(predicted_img)
        ax[1].set_title("Predicted Image")
        plt.show()

accuracy = accurate_predictions / total_predictions if total_predictions != 0 else 0
print(f"Accurate Predictions: {accurate_predictions}")
print(f"Total Predictions: {total_predictions}")
print(f"Accuracy: {accuracy * 100:.2f}%")

"""# Visual Search Test Holo"""

# Initialize predictions dict and other variables for accuracy calculation
test_dir = "/content/drive/MyDrive/visual_search_test_set/visual_search_test_set_holo"
accurate_predictions = 0
total_predictions = 0

for item_name in os.listdir(test_dir):
    item_path = os.path.join(test_dir, item_name)
    if os.path.isfile(item_path):
        total_predictions += 1
        vector_representation = get_vector_representation(item_path)

        # Query Pinecone index
        query_results = index.query(vector=[vector_representation.tolist()], top_k=1)
        if query_results.matches:
            predicted_class = query_results.matches[0].id.split('_')[0]
            actual_class = item_name.split('_')[0]
            print(f"Predicted class for {item_name}: {predicted_class}")

        # Increment correct prediction count if prediction is accurate
            if predicted_class == actual_class:
                accurate_predictions += 1

        # Displaying the actual vs. predicted images for all predictions
        actual_img = mpimg.imread(item_path)
        predicted_path = get_image_path(predicted_class)

        if predicted_path:
            predicted_img = mpimg.imread(predicted_path)
        else:
            print(f"No matching image found for predicted class: {predicted_class}")
            continue

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(actual_img)
        ax[0].set_title("Actual Image")
        ax[1].imshow(predicted_img)
        ax[1].set_title("Predicted Image")
        plt.show()

accuracy = accurate_predictions / total_predictions if total_predictions != 0 else 0
print(f"Accurate Predictions: {accurate_predictions}")
print(f"Total Predictions: {total_predictions}")
print(f"Accuracy: {accuracy * 100:.2f}%")

"""# VIsual Search Test Full Art

"""

# Initialize predictions dict and other variables for accuracy calculation
test_dir = "/content/drive/MyDrive/visual_search_test_set/visual_search_test_set_full_art"
accurate_predictions = 0
total_predictions = 0

for item_name in os.listdir(test_dir):
    item_path = os.path.join(test_dir, item_name)
    if os.path.isfile(item_path):
        total_predictions += 1
        vector_representation = get_vector_representation(item_path)

        # Query Pinecone index
        query_results = index.query(vector=[vector_representation.tolist()], top_k=1)
        if query_results.matches:
            predicted_class = query_results.matches[0].id.split('_')[0]
            actual_class = item_name.split('_')[0]
            print(f"Predicted class for {item_name}: {predicted_class}")

        # Increment correct prediction count if prediction is accurate
            if predicted_class == actual_class:
                accurate_predictions += 1

        # Displaying the actual vs. predicted images for all predictions
        actual_img = mpimg.imread(item_path)
        predicted_path = get_image_path(predicted_class)

        if predicted_path:
            predicted_img = mpimg.imread(predicted_path)
        else:
            print(f"No matching image found for predicted class: {predicted_class}")
            continue

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(actual_img)
        ax[0].set_title("Actual Image")
        ax[1].imshow(predicted_img)
        ax[1].set_title("Predicted Image")
        plt.show()

accuracy = accurate_predictions / total_predictions if total_predictions != 0 else 0
print(f"Accurate Predictions: {accurate_predictions}")
print(f"Total Predictions: {total_predictions}")
print(f"Accuracy: {accuracy * 100:.2f}%")