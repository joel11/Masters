#Pkg.add("SQLite")
using SQLite

function CreateDatabase(database_name)

    db = SQLite.DB(database_name)

    configuration_run_table =
    " CREATE TABLE IF NOT EXISTS configuration_run(
        configuration_id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_set_name VARCHAR,
        rbm_pretraining BOOL,
        sae_config_id INTEGER,
        seed_used INTEGER NOT NULL,
        start_time DATETIME NOT NULL)
        "

    network_parameters_table = "
        CREATE TABLE IF NOT EXISTS network_parameters(
            configuration_id INTEGER,
            category VARCHAR NOT NULL,
            layer_sizes VARCHAR NOT NULL,
            layer_activations VARCHAR NOT NULL,
            initialization VARCHAR NOT NULL)
            "
    training_parameters_table = "
        CREATE TABLE IF NOT EXISTS training_parameters(
            configuration_id INTEGER,
            category VARCHAR NOT NULL,
            learning_rate FLOAT NOT NULL,
            minibatch_size INTEGER NOT NULL,
            max_epochs INTEGER NOT NULL,
            l1_lambda FLOAT,
            l2_lambda FLOAT,
            cost_function VARCHAR,
            stopping_function VARCHAR)"

    epoch_records_table = "
        CREATE TABLE IF NOT EXISTS epoch_records(
            configuration_id INTEGER,
            category VARCHAR NOT NULL,
            record_time DATETIME NOT NULL,
            epoch_number INTEGER NOT NULL,
            mean_minibatch_cost FLOAT,
            training_cost FLOAT,
            testing_cost FLOAT,
            run_time FLOAT NOT NULL,
            mape FLOAT)"

    epoch_dataconfig_table = "
        CREATE TABLE IF NOT EXISTS dataset_config(
            configuration_id INTEGER,
            data_seed INTEGER,
            category VARCHAR NOT NULL,
            steps INTEGER NOT NULL,
            deltas VARCHAR NOT NULL,
            process_splits VARCHAR,
            training_splits VARCHAR,
            prediction_steps VARCHAR NOT NULL,
            variation_values VARCHAR)
    "

    prediction_results_table = "
        CREATE TABLE IF NOT EXISTS prediction_results(
            configuration_id INTEGER,
            time_step INTEGER,
            stock VARCHAR,
            actual FLOAT,
            predicted FLOAT)"

    SQLite.execute!(db, configuration_run_table)
    SQLite.execute!(db, training_parameters_table)
    SQLite.execute!(db, network_parameters_table)
    SQLite.execute!(db, epoch_records_table)
    SQLite.execute!(db, epoch_dataconfig_table)
    SQLite.execute!(db, prediction_results_table)
end

CreateDatabase("database_test.db")
