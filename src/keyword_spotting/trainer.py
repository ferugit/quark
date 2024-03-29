# Fernando López Gavilánez, 2023

import os
import torch
import numpy as np
import logging
import inspect
import time
import random

from sklearn.metrics import roc_curve, roc_auc_score, classification_report

import keyword_spotting.loader as loader


class AudioTrainer():

    def __init__(self, model, seed, num_classes, cuda, window_size, sampling_rate, 
        tensorboard_writer, reporter, results_path, model_id, batch_size):

        self._window_size = window_size
        self._sampling_rate = sampling_rate
        self._cuda = cuda
        self._set_seed(seed)
        self._model = model
        self._tensorboard_writer = tensorboard_writer
        self._reporter = reporter
        self._results_path = results_path
        self._model_id = model_id
        self.batch_size = batch_size

        # Select device
        if (torch.cuda.is_available() and cuda):
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._model.to(self._device)

    def prepare_data(self, partition_path, root_path):
        self.train_dataset, self.dev_dataset = loader.load_train_partitions(
            partition_path,
            root_path,
            window_size=int(self._window_size*self._sampling_rate)
            )
        self.test_dataset = loader.load_test_partition(
            partition_path,
            root_path,
            window_size=int(self._window_size*self._sampling_rate)
            )

    def train(self, epochs, optimizer, lr, momentum, weight_decay, balance, patience):

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.balance = balance
        self.patience = patience
        self.epochs = epochs

        logger.info(inspect.cleandoc(f'''Starting training:
            Epochs:             {self.epochs}
            Batch size:         {self.batch_size}
            Learning rate:      {self.lr}
            Training samples:   {self.train_dataset.n_samples}
            Validation samples: {self.dev_dataset.n_samples}
            Device:             {self._device.type}
            Optimizer:          {self.optimizer}
            Dataset classes:    {self.train_dataset.classes}
            Balance:            {self.balance}
            Patience:           {self.patience}
            Cuda:               {self._cuda}
        '''))

        # Optimizer
        if self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif(self.optimizer.lower() == 'rmsprop'):
            optimizer = torch.optim.RMSprop(self._model.parameters(),lr=self.lr, weight_decay=self.weight_decay)
        elif('sgd' in self.optimizer.lower()):
            optimizer = torch.optim.SGD(self._model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        # Loss function
        criterion = torch.nn.CrossEntropyLoss()

        # If balance dataset: use Weigth Random Sampler
        if self.balance:
            samples_per_class = 100 # empirical value
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.from_numpy(self.train_dataset.get_sample_weigths()),
                num_samples=self.train_dataset.n_classes*samples_per_class,
                replacement=True
                )
            shuffle_train = False
        else:
            sampler = None
            shuffle_train = True

        # Generate data loaders
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, 
            shuffle=shuffle_train, 
            batch_size=self.batch_size, 
            drop_last=False, 
            sampler=sampler
            )

        validation_loader = torch.utils.data.DataLoader(
            self.dev_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=False
            )

        # Metrics
        train_losses = []
        train_accuracies = []
        validation_losses = []
        validation_accuracies = []
        epoch_times = []
        
        # Early stopping
        best_loss_validation = np.inf
        patience_counter = self.patience

        # Get trainable parameters
        trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        logger.info('Number of trainable parameters: ' + str(trainable_params))

        # For present intermediate information
        n_intermediate_steps = int(len(train_loader)/3)
        
        # Start training loop
        print('Starting trainig...')
        for epoch in range(1, self.epochs+1):
            start_time = time.process_time()
            
            # Train model
            train_loss = 0.0
            train_accuracy = 0.0
            counter = 0

            self._model.train()
            for _, x, target in train_loader:
                counter += 1
                self._model.zero_grad()

                # Model forward
                out = self._model(x.to(self._device).float())

                # Backward and optimization
                loss = criterion(out, target.to(self._device))
                loss.backward()
                optimizer.step()

                # Store metrics
                train_loss += loss.item()
                train_accuracy += self._right_predictions(out, target)

                # Present intermediate results
                if (counter%n_intermediate_steps == 0):
                    logger.info("Epoch {}......Step: {}/{}....... Average Loss for Step: {} | Accuracy: {}".format(
                        epoch,
                        counter,
                        len(train_loader),
                        round(train_loss/counter, 4),
                        round(train_accuracy/(counter*self.batch_size), 4)
                        ))

            # Validate model
            validation_loss = 0.0
            validation_accuracy = 0.0

            with torch.no_grad():
                self._model.eval()
                for _, x, target in validation_loader:

                    out = self._model(x.to(self._device).float())
                    
                    # Store metrics: loss
                    loss = criterion(out, target.to(self._device))
                    validation_loss += loss.item()
                    validation_accuracy += self._right_predictions(out, target)
            
            # Calculate average losses
            train_loss = train_loss/len(train_loader)
            train_accuracy = train_accuracy/len(train_loader.sampler)
            validation_loss = validation_loss/len(validation_loader)
            validation_accuracy = validation_accuracy/len(validation_loader.sampler)
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)

            # Print epoch information
            current_time = time.process_time()
            logger.info("")
            logger.info("Epoch {}/{} Done.".format(epoch, self.epochs))
            logger.info("\t Tain Loss: {} |  Train Accuracy: {}".format(train_loss, train_accuracy))
            logger.info("\t Validation Loss: {} | Validation Accuracy: {}".format(validation_loss, validation_accuracy))
            logger.info("\t Time Elapsed for Epoch: {} seconds".format(str(current_time-start_time)))
            logger.info("")
            epoch_times.append(current_time-start_time)

            # Tensorboard
            self._tensorboard_writer.add_scalar("train_loss", train_loss, epoch)
            self._tensorboard_writer.add_scalar("train_accuracy", train_accuracy, epoch)
            self._tensorboard_writer.add_scalar("val_loss", validation_loss, epoch)
            self._tensorboard_writer.add_scalar("val_accuracy", validation_accuracy, epoch)
            self._tensorboard_writer.add_scalar("best_val_loss", best_loss_validation, epoch)
            self._tensorboard_writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
            self._tensorboard_writer.add_scalar("patience", patience_counter, epoch)
            self._tensorboard_writer.add_scalar("epoch_time", epoch_times[-1], epoch)

            # Early stopping
            if(best_loss_validation <= validation_loss):
                patience_counter += 1

                logger.info('Validation loss did not improve {:.3f} vs {:.3f}. Patience {}/{}.'.format(
                    validation_loss,
                    best_loss_validation,
                    patience_counter,
                    self.patience
                    ))
                logger.info("")
                
                if(patience_counter == self.patience):
                    logger.info('Breaking train loop: out of patience')
                    logger.info("")
                    break
            else:
                # Reinitialize patience counter and save model
                best_epoch = epoch
                patience_counter = 0
                best_loss_validation = validation_loss
                self._model.save_entire_model(os.path.join(self._results_path, self._model_id))

        logger.info("Total Training Time: {} seconds".format(str(sum(epoch_times))))

        # Report training
        metrics = {}
        metrics['training_time_s'] = sum(epoch_times)
        metrics['number_of_epochs'] = len(epoch_times)
        metrics['train_loss'] = train_losses
        metrics['validation_loss'] = validation_losses
        metrics['train_accuracy'] = train_accuracies
        metrics['validation_accuracy'] = validation_accuracies
        self._reporter.report("train_metrics", metrics)
        
        # Tensorboard
        self._tensorboard_writer.add_scalar("training_time", sum(epoch_times))
        self._tensorboard_writer.add_scalar("best_epoch", best_epoch)

    def test(self):

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Loss function
        criterion = torch.nn.CrossEntropyLoss()

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=False
            )

        # Metrics initialization
        test_loss = 0.0
        test_accuracy = 0.0

        labels = []
        predictions = []

        # For present intermediate information
        n_intermediate_steps = int(len(test_loader)/2)
        counter = 0

        with torch.no_grad():

            self._model.eval()
            
            for ID, x, target in test_loader:
                counter += 1

                # Model forward
                out = self._model(x.to(self._device).float())

                # Store metrics: loss and accuracy
                loss = criterion(out, target.to(self._device))
                test_loss += loss.item()
                test_accuracy += self._right_predictions(out, target)

                # Present intermediate results
                if (counter%n_intermediate_steps == 0):
                    logger.info("Epoch {}......Step: {}/{}....... Average Loss for Step: {} | Accuracy: {}".format(
                        1,
                        counter,
                        len(test_loader),
                        round(test_loss/counter, 4),
                        round(test_accuracy/(counter*self.batch_size), 4)
                        ))

                # Store labels and predictions
                labels += target.squeeze().tolist()
                predictions += out.squeeze().tolist()

            # Calculate average losses ans accuracies
            test_loss = test_loss/len(test_loader)
            test_accuracy = test_accuracy/len(test_loader.sampler)

            # Classification report
            target_names = []
            classes_index = self.test_dataset.class_index
            for index in range(len(classes_index)):
                target_names.append(classes_index[str(index)])
            report = self._get_metrics(labels, predictions, target_names=target_names)

            # Reporter
            metrics = {}
            metrics['test_loss'] = test_loss
            metrics['test_accuracy'] = test_accuracy
            metrics['report'] = report
            self._reporter.report('test_metrics', metrics)

            # Tensorboard test results
            self._tensorboard_writer.add_scalar("test_loss", metrics['test_loss'])
            self._tensorboard_writer.add_scalar("test_accuracy", metrics['test_accuracy'])
            #self._tensorboard_writer.add_scalar("macro-avg-f1-score", report[0]['macro avg']['f1-score'])
            # TODO: get confussion matrix and plot it
            #self._tensorboard_writer.add_figure("confussion_matrix", cm_fig)
            self._tensorboard_writer.close()

    def _right_predictions(self, out, label):
        """
        From a given set of labes and predictions 
        it uses a softmax function to determine the
        output label
        """
        bool_out = out.data.cpu() > 0.5
        bool_label = label > 0.5

        counter = 0
        for i in range(out.shape[0]):
            
            if (torch.all(torch.eq(bool_out[i], bool_label[i]))):
                counter += 1

        return counter

    def _get_metrics(self, labels, outputs, target_names=None, optimal_thresholds=None):
        """
        Get classification metrics
        Args:
            - labels: target labels
            - outputs: scores obtained from the model
            - target_names: used for the classification report
            - optimal_thersholds: if defined, thresholds aren't calculed
        Returns:
            - metrics: dict with metrics
            - opt_thresholds: optimal threshols
        """
        auc_list = []
        opt_thresholds = []
        labels_array = np.array(labels)
        outputs_array = np.array(outputs)
        n_classes = labels_array.shape[1]

        # Get AUC, optimal threshold, f1_score curve and  for each class
        for i in range(n_classes):
            true_labels, pred_scores = labels_array[:, i], outputs_array[:, i]
            fpr, tpr, thresholds, auc = self._compute_auc(true_labels, pred_scores)
            optimal_threshold, _ = self._get_youden_threshold(tpr, fpr, thresholds)
            auc_list.append(auc)
            opt_thresholds.append(optimal_threshold)
        
        # If there are already defined thresholds (arg) ignore the calculated ones
        if(optimal_thresholds):
            opt_thresholds = optimal_thresholds
        
        # Apply thresholds
        predicted_labels = self._scores_to_labels(outputs_array, opt_thresholds)

        # Classification Report
        report = classification_report(labels, predicted_labels, target_names=target_names, output_dict=True)

        # Add optimal threshold and AUC to metrics dict
        for index in range(n_classes):
            report[target_names[index]]['AUC'] = auc_list[index]
            report[target_names[index]]['Optimal_threshold'] = opt_thresholds[index]

        return report

    def _compute_auc(self, true_labels, pred_scores):
        """
        Computes the ROC AUC
        args:
            true_labels: target labels
            pred_scores: model output
        """
        fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
        roc_auc = roc_auc_score(true_labels, pred_scores)
        
        return fpr, tpr, thresholds, roc_auc

    def _get_youden_threshold(self, tpr, fpr, thresholds):
        "Calculates the optimal threshold using the Younden's index"
        J = tpr - fpr
        threshold_index = np.argmax(J)
        optimal_threshold = thresholds[threshold_index]
        return optimal_threshold, threshold_index

    def _scores_to_labels(self, outputs_array, thresholds):
        """
        Transform scores into labels using a different thershold per class.
        """
        predicted_labels = torch.zeros(outputs_array.shape).float()
        for i in range(outputs_array.shape[1]):
            predicted_labels[:, i] = (torch.tensor(outputs_array[:, i]) > thresholds[i]).float()
        
        return predicted_labels.data.numpy().tolist()

    def _set_seed(self, seed):
        """
        Fix seed of torch, numpy and random.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        random.seed(seed)

    def load_best_checkpoint(self):
        """
        Model attribute containes an overfitted version od the model.
        This method allows to load the best model in validation set.
        """
        self._model = torch.load(os.path.join(self._results_path, self._model_id + '_entire.pt'))