# Generated by Django 3.2.6 on 2023-07-24 15:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('home', '0005_rename_mask_image_uploadedimage_masked_image'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedimage',
            name='Volume',
            field=models.CharField(max_length=20, null=True),
        ),
    ]
