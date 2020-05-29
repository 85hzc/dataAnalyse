#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import wx

class MyApplication(wx.Frame):
	def __init__(self, parent, title):
		wx.Frame.__init__(self, parent, title=title, pos=(100, 200), size=(800,600))
		self.panel = wx.Panel(self)
		sizer = wx.BoxSizer(wx.VERTICAL)
		self.text1 = wx.TextCtrl(self.panel, value= "Hello wxPython", size=(400,300))
		sizer.Add(self.text1, 0, wx.ALIGN_TOP|wx.EXPAND)
		button = wx.Button(self.panel, label='click me')
		sizer.Add(button)
		self.panel.SetSizerAndFit(sizer)
		self.Bind(wx.EVT_BUTTON, self.OnClick, button)
		self.Show(True)
		self.panel.Bind(wx.EVT_MOVE,self.OnMove)
	def OnMove(self,e):
		posm = e.GetPosition()
		self.text1.Value = str(posm)
		
	def OnClick(self,e):
		self.text1.AppendText('\nHello wxpython!')
		
	

if __name__ == "__main__":
		app = wx.App()
		fr = MyApplication(None, "Hello wxPython")
		app.MainLoop()