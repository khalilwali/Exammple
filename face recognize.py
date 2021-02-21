import face_recognition
import os
from PIL import Image, ImageDraw


KNOWN_FACES_DIR = "images/known"
UNKNOWN_FACES_DIR = "images/unknown"
TOLERANCE = 0.6
MODEL = "hog"  # since my computer graphics card is weak. I'll be using the default hog model.


print("Loading known faces...")
known_faces = []
known_names = []

# We oranize known faces as subfolders of KNOWN_FACES_DIR
# Each subfolder's name becomes our label (name)
for name in os.listdir(KNOWN_FACES_DIR):

    # Next we load every file of faces of known person
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):

        # Load an image
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")

        # Get 128-dimension face encoding
        # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
        encoding = face_recognition.face_encodings(image)[0]

        # Append encodings and name
        known_faces.append(encoding)
        known_names.append(name)


print("Processing unknown faces...")
# Load image
image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/new1.jpg")

# This time we first grab face locations - we'll need them to draw boxes
locations = face_recognition.face_locations(image, model=MODEL)

# Now since we know loctions, we can pass them to face_encodings as second argument
# Without that it will search for faces once again slowing down whole process
encodings = face_recognition.face_encodings(image, locations)

# Convert to PIL format
pil_image = Image.fromarray(image)

# Create an ImageDraw Instance
draw = ImageDraw.Draw(pil_image)

# But this time we assume that there might be more faces in an image - we can find faces of dirrerent people
print(f", found {len(encodings)} face(s)")
for face_encoding, (top, right, bottom, left) in zip(encodings, locations):

    # We use compare_faces (but might use face_distance as well)
    # Returns array of True/False values in order of passed known_faces
    results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)

    # Since order is being preserved, we check if any face was found then grab index
    # then label (name) of first matching known face withing a tolerance
    name = "Unknown"
    if True in results:  # If at least one is true, get a name of first of found labels
        match = known_names[results.index(True)]
        print(f" - {match} found ")
        name = known_names[0]

    # Draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0))

    # Draw label
    text_width, text_height = draw.textsize(name)
    draw.rectangle(
        ((left, bottom - text_height - 10), (right, bottom)),
        fill=(255, 255, 0),
        outline=(255, 255, 0),
    )
    draw.text((left + 6, bottom - text_height - 5), name, fill=(0, 0, 0))

# it is always a good practice to delete the draw instance after using it.
del draw

# Display image
pil_image.show()

# Save image
pil_image.save("identify.jpg")
