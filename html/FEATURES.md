# Features and functionalities

<!-- TOC -->
* [Features and functionalities](#features-and-functionalities)
  * [Loading in Data](#loading-in-data)
    * [Selecting Loaded Data](#selecting-loaded-data)
    * [Information about the data](#information-about-the-data)
  * [Working with the data](#working-with-the-data)
    * [Changing the layout](#changing-the-layout)
    * [Getting more information](#getting-more-information)
      * [Indexes](#indexes)
      * [Individual read and write accesses vs total accesses](#individual-read-and-write-accesses-vs-total-accesses)
      * [Doing a deep dive into individual indexes](#doing-a-deep-dive-into-individual-indexes)
  * [Theme](#theme)
<!-- TOC -->

## Loading in Data

![Loading Data](https://user-images.githubusercontent.com/18115780/218279202-8f11fb70-ab89-4309-8600-d018c156e7b4.png)

Either when opening the HTML file for the first time, or by selecting None on the left sidebar the user is able to load in a JSON file.
It can either be drag and dropped into the gray area in the center, or alternatively you can click this area to then open a file selection dialog.

As described in the main [README](README.md) you can alternatively statically already include a single JSON file within a HTML file. But this will be slower due to parsing it in a different way. 

### Selecting Loaded Data

The data contained within the loaded JSON files can then be selected in the list on the left.

### Information about the data

Currently selected files do have some metadata displayed. This will be visible in the bottom of the sidebar.
By default, they will be collapsed, but the user can extend them to see the individual data. 

By hovering the address range in the *Active Memory Information* section you can also see how many elements this specific array contained, and what the size of a single element was.

This is shown in the screenshot below.

![image](https://user-images.githubusercontent.com/18115780/218279550-3d6f6313-17db-4147-a559-d6e9f28b441c.png)

## Working with the data

### Changing the layout

A key feature is changing the layout. With defaults settings all the data is just fit on screen. But often data has meaning, especially in 2D data, or for example the reduction which is used as an example here.

For this "width" of a row can be set on the bottom left. This would look like the following in the reduction example:

![image](https://user-images.githubusercontent.com/18115780/218279692-fea204c7-6071-4686-9ef7-1b9f0d33f30a.png)

After setting it to a certain width, the user can pan left and right. (Additionally to up and down)

Although it is rarely needed, there is a dedicated button to change from a grid layout (default) to one long list.

![image](https://user-images.githubusercontent.com/18115780/218279757-fdf40457-7d5c-469c-9d52-598a2731278e.png)

This is (besides an implementation detail) the same as setting the width to 1 on the bottom left. 

But this a good example to explain the meaning of the buttons. **__The icon of a button (besides the light mode toggle) always represents the state change which would happen.__** So for example if you are in grid mode, the icon will be a linearized icon, if you are in linear mode, the icon will be a grid.

### Getting more information

#### Indexes

By default, no index information is displayed, to have a "cleaner" look. If you want to know the index of an individual element, you can hover it for a bit, and then it's index will be displayed. But should this not be sufficient, you can toggle that indices are always displayed.

See the image below as example:

![image](https://user-images.githubusercontent.com/18115780/218279988-c6280236-ae8c-47a8-a7a3-c78f430aa945.png)

#### Individual read and write accesses vs total accesses

With unchanged settings, read and write accesses will be shown individually. Then each row contains in blue the read accesses on top, and in orange the write accesses below.

The behavior can be toggled with the button shown in the image below.

![imgage](https://user-images.githubusercontent.com/18115780/218280884-6a6965f8-5ccb-41f7-a0c5-0196f7688bbf.png)

As you might already have spotted, total accesses additionally has blue color palette button in the header bar.
This is because for total accesses you might care especially about certain patterns, or even slight differences.
To make this easier, it is possible to adjust the color mapping function which is used to color the different values.

For that click the colored icon in the top bar and you will be greeted with the following menu:

![image](https://user-images.githubusercontent.com/18115780/218280217-a0e64719-c89a-4516-9dba-2e1bb7da03ec.png)

In this menu you can enable the use of a cube helix color mapping function, and then additionally change all the parameters in the interface, while seeing a preview of how the function would look like.

On the bottom you will also find a few presets, which also allow you to quickly reset the values to a default scheme.

To make comparisons between different data structures easier, it is also possible to set a custom reference maximum value. By default, the maximum value of the displayed data is used, but you could also set this to a higher or lower value, to see if one algorithm (performing the same tasks) does better or worse considering memory accesses. 

Using 100 memory accesses as reference, instead of the 11 which the reduced example contains would look like this:

![image](https://user-images.githubusercontent.com/18115780/218280355-0d492674-fe1c-46ac-8512-a3ee389cd69d.png)

#### Doing a deep dive into individual indexes

To get more specific information, instead of an overview, it is also possible to select a single index (*You have to click the cell with the number representing the accesses, not the index itself if you are displaying the indexes*) by clicking on it.

This will then open up a drawer which shows you the individual accesses with their respective thread ids and block ids in chronological order.

By default, the drawer will only show the data of what you clicked on (read only, write only, total), but you can also toggle this mode within the drawer by clicking the header.

The drawer looks like the following:

![image](https://user-images.githubusercontent.com/18115780/218280516-c8bc3f48-0016-4503-979d-3754d16f762b.png)

## Theme

For people preferring light mode (for example for better readability), this is actually included in the app.

In the top right next to the *About* button, you have two toggles. The Switch type button toggles the theme of the entire app, the button with the lamp icon toggles the background color of the cells.

Toggling both looks like the following:

![image](https://user-images.githubusercontent.com/18115780/218280714-1deddf62-9aed-4721-8793-97c66795ad9b.png)
