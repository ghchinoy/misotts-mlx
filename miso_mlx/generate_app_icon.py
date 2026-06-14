import os
import sys
import math
from PIL import Image, ImageDraw, ImageFilter

def create_master_icon(output_path):
    size = 1024
    # Create transparent image
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # 1. Draw modern macOS squircle/rounded-rect container
    # Standard macOS Big Sur app icon grid dictates a centered 824x824 squircle within 1024x1024 for shadow room.
    margin = 100
    rect_size = size - 2 * margin # 824x824
    
    # We will draw a premium gradient background inside the squircle
    # Since Pillow doesn't do complex gradients natively, we can draw line-by-line or generate a gradient array
    bg = Image.new("RGBA", (rect_size, rect_size), (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg)
    
    # Generate high-end dark gradient (Deep Purple to Sleek Indigo/Dark Blue)
    for y in range(rect_size):
        ratio = y / rect_size
        # Interpolate between Deep Purple (24, 18, 48) and Sleek Dark Teal/Blue (15, 32, 67)
        r = int(24 + ratio * (15 - 24))
        g = int(18 + ratio * (32 - 18))
        b = int(48 + ratio * (67 - 48))
        bg_draw.line([(0, y), (rect_size, y)], fill=(r, g, b, 255))
        
    # Apply rounded corner mask to background
    mask = Image.new("L", (rect_size, rect_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    # macOS squircle corner radius is ~180px for a 824x824 canvas
    mask_draw.rounded_rectangle([0, 0, rect_size, rect_size], radius=180, fill=255)
    
    # Composite the gradient into a rounded rect
    squircle = Image.new("RGBA", (rect_size, rect_size), (0, 0, 0, 0))
    squircle.paste(bg, (0, 0), mask=mask)
    
    # Draw standard macOS squircle border/stroke (subtle premium highlights)
    border_mask = Image.new("L", (rect_size, rect_size), 0)
    border_mask_draw = ImageDraw.Draw(border_mask)
    border_mask_draw.rounded_rectangle([0, 0, rect_size, rect_size], radius=180, fill=255)
    
    border_inner = Image.new("L", (rect_size, rect_size), 255)
    border_inner_draw = ImageDraw.Draw(border_inner)
    border_inner_draw.rounded_rectangle([2, 2, rect_size - 2, rect_size - 2], radius=178, fill=0)
    
    stroke_mask = Image.new("L", (rect_size, rect_size), 0)
    stroke_mask.paste(border_mask, (0, 0), mask=border_inner)
    
    # Paste a subtle white-silver stroke highlighting the squircle edge
    stroke_color = Image.new("RGBA", (rect_size, rect_size), (255, 255, 255, 40))
    squircle.paste(stroke_color, (0, 0), mask=stroke_mask)
    
    # 2. Draw glowing aesthetic soundwave/voice design in the center
    # We will draw several beautiful glowing bezier/sine-like curves in the center representing speech synthesis
    wave_img = Image.new("RGBA", (rect_size, rect_size), (0, 0, 0, 0))
    wave_draw = ImageDraw.Draw(wave_img)
    
    center_y = rect_size // 2 + 10
    width = rect_size
    
    # Let's draw 3 waves with different colors, amplitudes, and offsets
    # Neon Pink/Purple wave, Cyan/Blue wave, Orange/Yellow accent wave
    waves = [
        # (color, amplitude, freq, phase, stroke_width, alpha)
        ((255, 50, 150), 110, 0.015, 0.0, 14, 230),
        ((0, 180, 255), 80, 0.022, 1.8, 10, 180),
        ((255, 130, 0), 50, 0.035, 3.5, 6, 150)
    ]
    
    for color, amp, freq, phase, stroke, alpha in waves:
        points = []
        # We fade amplitudes at the edges using a Gaussian-like window to look clean
        for x in range(40, rect_size - 40):
            # Normalised x coordinate -0.5 to 0.5
            nx = (x - rect_size/2) / (rect_size/2 - 40)
            # Gaussian window to damp the wave at the edges
            window = math.exp(-3.5 * nx**2)
            
            y = center_y + amp * math.sin(freq * x + phase) * window
            points.append((x, y))
            
        # Draw the wave lines
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            wave_draw.line([p1, p2], fill=color + (alpha,), width=stroke)
            
    # Apply a subtle blur to the wave image to create a gorgeous glow effect
    glow_img = wave_img.filter(ImageFilter.GaussianBlur(12))
    
    # Double-composite the waves to make them stand out
    squircle.paste(glow_img, (0, 0), mask=glow_img)
    squircle.paste(wave_img, (0, 0), mask=wave_img)
    
    # 3. Paste squircle container onto the main transparent canvas (with shadow room)
    # Draw a soft drop-shadow behind the squircle
    shadow_mask = Image.new("L", (size, size), 0)
    shadow_mask_draw = ImageDraw.Draw(shadow_mask)
    shadow_mask_draw.rounded_rectangle([margin, margin + 8, margin + rect_size, margin + rect_size + 8], radius=180, fill=120)
    shadow = shadow_mask.filter(ImageFilter.GaussianBlur(24))
    
    shadow_color = Image.new("RGBA", (size, size), (0, 0, 0, 100))
    image.paste(shadow_color, (0, 0), mask=shadow)
    
    # Paste the squircle on top
    image.paste(squircle, (margin, margin), mask=squircle)
    
    # Save the master image
    image.save(output_path, "PNG")
    print(f"✅ Master 1024x1024 app icon generated at: {output_path}")

if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "master_icon.png"
    create_master_icon(out_file)
