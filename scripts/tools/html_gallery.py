

html_template = '<a target="_blank" href="{}"><img src="{}" width="{}" height="{}"></a>'



def get_gallery(images, width, height):
    out_html = ''
    n = 0
    for image in images:
        # provide variable as an argument in the format() method
        out_html += html_template.format(image, image, width, height)

    return out_html


def createHTMLGallery(images):
    html3 = """
        <div class="gallery-history" style="
    display: flex;
    flex-wrap: wrap;
	align-items: flex-start;">
        """
    mkdwn_array = []
    for i in images:
        bImg = i.read()
        i = Image.save(bImg, 'PNG')
        width, height = i.size
        #get random number for the id
        image_id = "%s" % (str(images.index(i)))
        (data, mimetype) = STImage._normalize_to_bytes(bImg.getvalue(), width, 'auto')
        this_file = in_memory_file_manager.add(data, mimetype, image_id)
        img_str = this_file.url
        #img_str = 'data:image/png;base64,' + b64encode(image_io.getvalue()).decode('ascii')
        #get image size

        #make sure the image is not bigger then 150px but keep the aspect ratio
        if width > 150:
            height = int(height * (150/width))
            width = 150
        if height > 150:
            width = int(width * (150/height))
            height = 150

        #mkdwn = f"""<img src="{img_str}" alt="Image" with="200" height="200" />"""
        mkdwn = f'''<div class="gallery" style="margin: 3px;" >
<a href="{img_str}">
<img src="{img_str}" alt="Image" width="{width}" height="{height}">
</a>
</div>
'''
        mkdwn_array.append(mkdwn)
    html3 += "".join(mkdwn_array)
    html3 += '</div>'
    return html3