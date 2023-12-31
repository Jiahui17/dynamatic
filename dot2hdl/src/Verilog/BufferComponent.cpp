/*
 * BufferComponent.cpp
 *
 *  Created on: 17-Jun-2021
 *      Author: madhur
 */


#include "ComponentClass.h"

//Subclass for Entry type component
BufferComponent::BufferComponent(Component& c){
	index = c.index;
	if(c.type == COMPONENT_BUF)
		moduleName = "elasticBuffer";
	else if(c.type == COMPONENT_TEHB)
		moduleName = "TEHB";
	else if(c.type == COMPONENT_OEHB)
			moduleName = "OEHB";
	name = c.name;
	instanceName = name;
	type = c.type;
	bbID = c.bbID;
	op = c.op;
	in = c.in;
	out = c.out;
	delay = c.delay;
	latency = c.latency;
	II = c.II;
	slots = c.slots;
	transparent = c.transparent;
	value = c.value;
	io = c.io;
	inputConnections = c.inputConnections;
	outputConnections = c.outputConnections;

	clk = c.clk;
	rst = c.rst;
}
